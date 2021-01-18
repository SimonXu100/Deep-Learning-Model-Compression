import os
from train_ResNet18 import *

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.utils.prune as prune
import torch.nn.functional as F

import argparse

parser = argparse.ArgumentParser(
    description='PyTorch AlexNet Prune & Retraining')
parser.add_argument('--prune_fraction', type=float, default=0.5,
                    help='Fraction of parameters to prune each iteration')
parser.add_argument('--iterations', type=int, default=6,
                    help='Number of iterations for iterative pruning')
parser.add_argument('--random_prune', action='store_true',
                    help='Randomly select connections to prune')


def finetune(model, num_epochs, trainloader, testloader, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    test_accus = train(model, device, trainloader, testloader, criterion,
                       optimizer, num_epochs, None, scheduler=None)
    train_accu = evaluate(model, trainloader, device)
    print("Finetune test accus:", test_accus)
    return test_accus[-1], train_accu


def get_mask_sum(module):
    mask_sum = 0
    children = list(module.named_children())
    if len(children) > 0:
        for name, module in children:
            mask_sum += get_mask_sum(module)
    else:
        for name, mask in module.named_buffers():
            mask_sum += mask.sum().item()
    return mask_sum


def get_parameters_to_prune(module, attrs2prune, module_name="model"):
    parameters_to_prune = []
    parameters_names = []
    children = list(module.named_children())
    if len(children) > 0:
        for name, module in children:
            output = get_parameters_to_prune(module, attrs2prune, module_name=module_name+"."+name)
            parameters_to_prune += output[0]
            parameters_names += output[1]
    else:
        for name, param in module.named_parameters():
            if name in attrs2prune:
                parameters_to_prune.append((module, name))
                parameters_names.append(module_name+"."+name)
    return parameters_to_prune, parameters_names


if __name__ == "__main__":
    MODEL_PATH = 'models/resnet18_finetuned.pth'
    CHKPT_DIR = "resnet18_chkpt"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    print(args)

    batch_size = 64
    prune_iteration = args.iterations
    prune_fraction = args.prune_fraction

    np.random.seed(0)
    torch.manual_seed(0)

    if args.random_prune:
        pruning_method = prune.RandomUnstructured
    else:
        pruning_method = prune.L1Unstructured

    # trainset, testset = load_cifar10_pytorch(root='G:\ML dataset', transform=ImageNet_Transform_Func)
    trainset, testset = load_cifar10_pytorch(transform=ImageNet_Transform_Func)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
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
    
    get_parameters_to_prune(model, ("weight", "bias"))
    parameters_to_prune, parameters_names = get_parameters_to_prune(model, ("weight", "bias"))

    print(pruning_method)
    for i in range(prune_iteration):
        print("=========================Iteration %i =========================="%(i+1))
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=pruning_method,
            amount=prune_fraction,
        )
        frac_list.append(frac_list[-1]*(1-prune_fraction))

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

        sum_zero_weight = 0
        sum_weight = 0
        for module, parameter in parameters_to_prune:
            cur_zero_weight = float(torch.sum(getattr(module,parameter) == 0))
            cur_weight = float(getattr(module,parameter).nelement())
            print("Sparsity in {}.{}: {:.2f}%".format(module, parameter,
                100. * cur_zero_weight/cur_weight))
            sum_zero_weight+=cur_zero_weight
            sum_weight += cur_weight
        print("Global sparsity: {:.2f}%".format(100. * sum_zero_weight/sum_weight))
        print(sum_zero_weight, sum_weight)

    result = np.vstack((frac_list, test_accus_prune, train_accus_prune, test_accus_prune_finetuned, train_accus_prune_finetuned))
    np.savetxt("resnet18_unstructured_performance.txt", result)