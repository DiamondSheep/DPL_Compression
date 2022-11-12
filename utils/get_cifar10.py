import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

CIFAR10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_TRAIN_STD = (0.2023, 0.1994, 0.2010)

CIFAR10_TEST_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_TEST_STD = (0.2023, 0.1994, 0.2010)

def get_training_dataloader(data_path, batch_size=16, num_workers=0, shuffle=True):

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)
    ])
    #cifar10_training = cifar10Train(path, transform=transform_train)
    cifar10_training = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    cifar10_training_loader = DataLoader(
        cifar10_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_training_loader

def get_test_dataloader(data_path, batch_size=16, num_workers=0, shuffle=True):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_TEST_MEAN, CIFAR10_TEST_STD)
    ])
    #cifar10_test = cifar10Test(path, transform=transform_test)
    cifar10_test = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
    cifar10_test_loader = DataLoader(
        cifar10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_test_loader

classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')
