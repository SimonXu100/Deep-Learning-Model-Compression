import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms

import argparse

parser = argparse.ArgumentParser(
    description='PyTorch ResNet50 training')

parser.add_argument('--freeze', action='store_true',
                    help='Don\'t prune feature extractor of AlexNet')


def save_chkpt(model, epoch, val_acc, chkpt_dir):
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)

    state = {'model_state_dict': model.state_dict(),
             'epoch': epoch,
             'val_acc': val_acc, }
    torch.save(state, "%s/checkpoint_%i_%.2f.pth" % (chkpt_dir, epoch, val_acc))
    print("checkpoint saved at epoch", epoch)


def load_chkpt(model, chkpt_path, device):
    checkpoint = torch.load(chkpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    val_acc = checkpoint['val_acc']
    print("epoch, val_acc:", epoch, val_acc)
    return model


def load_cifar10_pytorch(root='./data',
                         transform=transforms.Compose([transforms.ToTensor()])):
    train_set = torchvision.datasets.CIFAR10(root=root, train=True,
                                             download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False,
                                            transform=transform)
    return train_set, test_set


def train(model, device, train_loader, test_loader, criterion,
          optimizer, max_epoch, chkpt_dir, scheduler = None):
    test_accus = []
    for epoch in range(max_epoch):
        model.train()
        for i, data in enumerate(train_loader):
            xs, ys = data
            xs, ys = xs.to(device), ys.to(device)
            outputs = model(xs)

            loss = criterion(outputs, ys)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                _, predicted = torch.max(outputs, 1)
                accu = (predicted == ys).double().sum().item() / ys.shape[
                    0] * 100
                print('\r', "Epoch:", epoch + 1, "Iter:", i + 1, "loss =",
                      loss.data.item(), "accu =", "{:.2f}%".format(accu),
                      end='')
        if scheduler:
            scheduler.step()

        print('')
        cur_test_accu = evaluate(model, test_loader, device)
        print("Test accu = {:.2f}%".format(cur_test_accu * 100))
        if chkpt_dir and (len(test_accus) == 0 or cur_test_accu > np.max(test_accus)):
            save_chkpt(model, epoch, cur_test_accu, chkpt_dir)
        test_accus.append(cur_test_accu)

    return test_accus


def evaluate(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        num_correct = 0
        for i, data in enumerate(test_loader):
            xs, ys = data
            xs, ys = xs.to(device), ys.to(device)
            outputs = model(xs)
            _, predicted = torch.max(outputs, 1)
            num_correct += (predicted == ys).double().sum().item()
        test_accu = num_correct / len(test_loader.dataset)
        return test_accu


def set_parameter_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad

ImageNet_Transform_Func = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])



if __name__ == "__main__":
	args = parser.parse_args()

	batch_size = 128
	max_epoch = 20
	
	# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	trainset, testset = load_cifar10_pytorch(transform=ImageNet_Transform_Func)
	# trainset, testset = load_cifar10_pytorch()
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
	                                      shuffle=True, num_workers=2)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
	                                     shuffle=False, num_workers=2)

	classes = ('plane', 'car', 'bird', 'cat',
	       'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	if args.freeze:
		# only train the last fc layer
		model = models.resnet50(pretrained=True)
		set_parameter_requires_grad(model, False)

		chkpt_dir = "resnet50_chkpt"
		num_ftrs = model.fc.in_features
		model.fc = nn.Linear(num_ftrs, 10)

		model.to(DEVICE)
		criterion = torch.nn.CrossEntropyLoss()
		optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9)
		scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
		test_accus = train(model, DEVICE, trainloader, testloader, criterion,
		               optimizer, max_epoch, chkpt_dir, scheduler=scheduler)
		print(test_accus)
	else:
		model = models.resnet50(pretrained=True)
		set_parameter_requires_grad(model, True)

		chkpt_dir = "resnet50_chkpt_whole"
		num_ftrs = model.fc.in_features
		model.fc = nn.Linear(num_ftrs, 10)

		model.to(DEVICE)
		criterion = torch.nn.CrossEntropyLoss()
		optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
		scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
		test_accus = train(model, DEVICE, trainloader, testloader, criterion,
		               optimizer, max_epoch, chkpt_dir, scheduler=scheduler)
		print(test_accus)

