from train_AlexNet import *

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.utils.prune as prune

import argparse

parser = argparse.ArgumentParser(
    description='PyTorch AlexNet Prune & Retraining')
parser.add_argument('--prune_fraction', type=float, default=0.1,
                    help='Fraction of parameters to prune each iteration')
parser.add_argument('--iterations', type=int, default=6,
                    help='Number of iterations for iterative pruning')


def finetune(model, num_epochs, trainloader, testloader, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0003, momentum=0.9)
    test_accus = train(model, device, trainloader, testloader, criterion,
                       optimizer, num_epochs, None, scheduler=None)
    train_accu = evaluate(model, trainloader, device)
    print("Finetune test accus:", test_accus)
    return test_accus[-1], train_accu


def print_mask_sum(root_module):
    mask_sum = 0
    for name, module in root_module.named_children():
        for name, mask in module.named_buffers():
            mask_sum += mask.sum().item()
    print("Number of not-masked paramters:", mask_sum)
    return mask_sum


def print_module_weights(root_module):
    for name,module in root_module.named_children():
        print(name)
        if hasattr(module, "weight"):
            print(module.weight)


def get_conv_modules(module):
    conv_modules = []
    for name, module in module.named_children():
        if isinstance(module, nn.Conv2d):
            conv_modules.append(module)
    return conv_modules


def get_num_params(conv_modules):
    num_params_conv = 0
    for module in conv_modules:
        num_params_conv += module.weight.nelement()
    return num_params_conv


if __name__ == "__main__":
    MODEL_PATH = './alexnet_finetuned.pth'
    CHKPT_DIR = "alexnet_chkpt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    print(args)

    batch_size = 128
    prune_iteration = args.iterations
    prune_fraction = args.prune_fraction

    np.random.seed(0)
    torch.manual_seed(0)

    # trainset, testset = load_cifar10_pytorch(root='G:\ML dataset', transform=ImageNet_Transform_Func)
    trainset, testset = load_cifar10_pytorch(transform=ImageNet_Transform_Func)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    model = models.alexnet()
    model.classifier[6] = nn.Linear(4096, 10)
    model.to(DEVICE)
    print(model)
    load_chkpt(model, MODEL_PATH, DEVICE)

    test_accu, train_accu = evaluate(model,testloader, DEVICE), evaluate(model,trainloader, DEVICE)
    print(test_accu, train_accu)

    frac_list = [100]
    test_accus_prune= [test_accu]
    train_accus_prune= [train_accu]
    test_accus_prune_finetuned = [test_accu]
    train_accus_prune_finetuned = [train_accu]
    # We are only structurally pruning the convolutional layers
    conv_modules = get_conv_modules(model.features)
    print(conv_modules)
    num_params_conv = get_num_params(conv_modules)
    print("Number of weight parameters in all conv layers:", num_params_conv)
    for i in range(prune_iteration):
        print("=========================Iteration %i =========================="%(i+1))
        for module in conv_modules:
            prune.ln_structured(module, "weight", amount=prune_fraction, n=2,dim=0)
        frac_list.append(frac_list[-1]*(1-prune_fraction))

        print_mask_sum(model.features)
        test_accu, train_accu = evaluate(model, testloader, DEVICE), evaluate(
            model, trainloader, DEVICE)
        print("Performance before finetuning:")
        print("Test accuracy:", test_accu)
        print("Training accuracy:", train_accu)
        test_accus_prune.append(test_accu)
        train_accus_prune.append(train_accu)

        test_accu, train_accu = finetune(model, 5, trainloader, testloader, DEVICE)
        print("Performance after finetuning:")
        print("Test accuracy:", test_accu)
        print("Training accuracy:", train_accu)
        test_accus_prune_finetuned.append(test_accu)
        train_accus_prune_finetuned.append(train_accu)
        torch.save(model, "alexnet_chkpt/model_conv_frac_%.2f.pth"%frac_list[-1])

    result = np.vstack((frac_list, test_accus_prune, train_accus_prune, test_accus_prune_finetuned, train_accus_prune_finetuned))
    np.savetxt("Alexnet_structured_performance.txt", result)

