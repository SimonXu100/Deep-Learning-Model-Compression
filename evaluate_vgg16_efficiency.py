from prune_structured_vgg16 import *

import torch
import torch.nn as nn
import torchvision.models as models

import torch_pruning as pruning


def get_average_inference_time(model, device, batch_size_list):
    model.eval()
    with torch.no_grad():
        # do a inference to make sure the model is loaded in GPU
        _ = model(torch.randn(1, 3, 224, 224).to(device))

        result = []
        for batch_size in batch_size_list:
            inference_time_sum = 0
            for i in range(10):
                x = torch.randn(batch_size, 3, 224, 224).to(device)

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(x)
                end.record()
                torch.cuda.synchronize()

                cur_inference_time = start.elapsed_time(end)
                inference_time_sum += cur_inference_time
            result.append(inference_time_sum/10)

    return result


def finetune_bias(model, num_epochs, trainloader, testloader, device):
    params_to_train = []
    # Since the difference between the pytorch pruned model and the actual
    # pruned model is the bias of the pruned layers. We can restore the
    # accuracy by only finetune the bias of the remaining filters
    for name, param in model.named_parameters():
        if "bias" in name:
            params_to_train.append(param)

    criterion = torch.nn.CrossEntropyLoss()
    # lr should be adjusted based on the remained fraction of the network
    # Some tested values: 90%: 0.005, 53.14%:0.05, 34.87%:0.1
    optimizer = torch.optim.SGD(params_to_train, lr=0.02, momentum=0.9)
    test_accus = train(model, device, trainloader, testloader, criterion,
                       optimizer, num_epochs, None, scheduler=None)
    print("Finetune test accus:", test_accus)

    return np.max(test_accus)


if __name__ == "__main__":
    batch_size = 128
    # For testing inference time
    batch_size_list = [1, 4, 16, 64, 256]
    DEVICE = "cuda:0"
    MODEL_PATH = 'models/vgg16_finetuned.pth'
    # Set this flag to true if you want to check the accuracy of pruned and original model
    # It will also finetune bias of the actually pruned model
    CHECK_ACCURACY = False
    FRAC_LIST = [90]
    for i in range(9):
        FRAC_LIST.append(FRAC_LIST[-1] * 0.9)

    num_params_list = [FRAC_LIST, [],[],[]]
    for frac in FRAC_LIST:
        print("==================== Current Fraction ====================")
        print("")
        print("%.2f"%frac)
        print("")
        print("==========================================================")
        PRUNED_MODEL_PATH = "vgg16_chkpt/model_conv_frac_%.2f.pth"%frac

        # trainset, testset = load_cifar10_pytorch(root='G:\ML dataset',
        #                                          transform=ImageNet_Transform_Func)
        trainset, testset = load_cifar10_pytorch(transform=ImageNet_Transform_Func)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)

        inference_time_matrix = [batch_size_list]
        print("==================== Original Model ====================")
        model = models.vgg16()
        model.classifier[6] = nn.Linear(4096, 10)
        model.to(DEVICE)
        load_chkpt(model, MODEL_PATH, DEVICE)

        conv_modules = get_conv_modules(model.features)
        num_params = get_num_params(conv_modules)
        print("Number of params:", num_params)
        num_params_list[1].append(num_params)
        if CHECK_ACCURACY:
            print("Test Accuracy", evaluate(model, testloader, DEVICE))

        average_inference_time = get_average_inference_time(model, DEVICE, batch_size_list)
        print("Batch size list for evaluating inference time:", batch_size_list)
        print("Average Inference time in ms:", average_inference_time)
        inference_time_matrix.append(average_inference_time)
        print("==================== Model Pruned with Pytorch (with mask) ====================")
        model_pruned = torch.load(PRUNED_MODEL_PATH)
        model_pruned.to(DEVICE)
        conv_modules = get_conv_modules(model_pruned.features)
        print("Number of params:", get_num_params(conv_modules))
        # Here we use mask sum as number of parameters since this is the number of
        # effective parameters pytorch think there it is.
        num_params = print_mask_sum(model_pruned.features)
        num_params_list[2].append(num_params)
        print("Model:")
        print(model_pruned)
        if CHECK_ACCURACY:
            print("Test Accuracy", evaluate(model_pruned, testloader, DEVICE))

        average_inference_time = get_average_inference_time(model_pruned, DEVICE, batch_size_list)
        print("Batch size list for evaluating inference time:", batch_size_list)
        print("Average Inference time in ms:", average_inference_time)
        inference_time_matrix.append(average_inference_time)

        # This only make the prune permanant, the size of the model remains the same
        # The pruned weight are set to be zero. This won't affect the inference time
        # This is required to make the pruning package work.
        for module in conv_modules:
            prune.remove(module, 'weight')
        print("Number of params after removing mask:", get_num_params(conv_modules))
        for module in conv_modules:
            filter_sum_weight = module.weight.sum(dim=(1,2,3)).detach().cpu().numpy()
            pruning_idxs = np.where(filter_sum_weight==0)[0].tolist()
            DG = pruning.DependencyGraph()
            DG.build_dependency(model_pruned, example_inputs=torch.randn(1, 3, 224, 224))
            pruning_plan = DG.get_pruning_plan(module, pruning.prune_conv, idxs = pruning_idxs)
            pruning_plan.exec()

        print("==================== Actual pruned Model ====================")
        num_params = get_num_params(conv_modules)
        print("Number of params:", num_params)
        num_params_list[3].append(num_params)
        print("Model:")
        print(model_pruned)

        model_pruned.to(DEVICE)
        if CHECK_ACCURACY:
            print("Actual Pruned Test Accuracy before finetuning:", evaluate(model_pruned, testloader, DEVICE))
            test_accu = finetune_bias(model_pruned, 5, trainloader, testloader, DEVICE)
            print("Actual Pruned Test Accuracy before finetuning:", test_accu)

        average_inference_time = get_average_inference_time(model_pruned, DEVICE, batch_size_list)
        print("Batch size list for evaluating inference time:", batch_size_list)
        print("Average Inference time in ms:", average_inference_time)
        inference_time_matrix.append(average_inference_time)

        np.savetxt("performance/vgg16_structured_%.2f_efficiency.txt"%frac, inference_time_matrix)


    np.savetxt("performance/vgg16_structured_num_params.txt", num_params_list)