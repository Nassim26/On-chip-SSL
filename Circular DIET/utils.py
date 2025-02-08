import torch
import numpy as np
from torch.utils.data import Dataset, Subset
import torchvision

def save_logs(log_data, filename="training_logs.txt"):
    with open(filename, "w") as file:
        file.writelines(log_data)

def save_model(model, path):
    torch.save(model.state_dict(), path)

class DatasetWithIndices(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.permutation = np.random.permutation(len(dataset))

    def __getitem__(self, n):
        x, y = self.dataset[n]
        n = torch.tensor([self.permutation[n]])
        return x, y, n

    def __len__(self):
        return len(self.dataset)

def get_datasets(config):
    if config.dataset == "MNIST":
        training_data = torchvision.datasets.MNIST(
            train=True, download=True, root="./data",
            transform=torchvision.transforms.Compose(config.train_transform)
        )
        test_data = torchvision.datasets.MNIST(
            train=False, download=True, root="./data",
            transform=torchvision.transforms.Compose(config.test_transform)
        )
    elif config.dataset == "CIFAR10":
        training_data = torchvision.datasets.CIFAR10(
            train=True, download=True, root="./data",
            transform=torchvision.transforms.Compose(config.train_transform)
        )
        test_data = torchvision.datasets.CIFAR10(
            train=False, download=True, root="./data",
            transform=torchvision.transforms.Compose(config.test_transform)
        )
    elif config.dataset == "CIFAR100":
        training_data = torchvision.datasets.CIFAR100(
            train=True, download=True, root="./data",
            transform=torchvision.transforms.Compose(config.train_transform)
        )
        test_data = torchvision.datasets.CIFAR100(
            train=False, download=True, root="./data",
            transform=torchvision.transforms.Compose(config.test_transform)
        )
    elif config.dataset == "KMNIST":
        training_data = torchvision.datasets.KMNIST(
            train=True, download=True, root="./data",
            transform=torchvision.transforms.Compose(config.train_transform)
        )
        test_data = torchvision.datasets.KMNIST(
            train=False, download=True, root="./data",
            transform=torchvision.transforms.Compose(config.test_transform)
        )
    elif config.dataset == "FashionMNIST":
        training_data = torchvision.datasets.FashionMNIST(
            train=True, download=True, root="./data",
            transform=torchvision.transforms.Compose(config.train_transform)
        )
        test_data = torchvision.datasets.FashionMNIST(
            train=False, download=True, root="./data",
            transform=torchvision.transforms.Compose(config.test_transform)
        )
    else:
        print(f"Dataset '{config.dataset}' is not implemented.")
        raise NotImplementedError

    if config.limit_data < float('inf'):
        indices = torch.arange(config.limit_data)
        training_data = Subset(training_data, indices)

    training_data = DatasetWithIndices(training_data)

    total_size = len(training_data)
    train_enc_size = int(0.7 * total_size)
    train_cls_size = int(0.2 * total_size)
    test_size = total_size - train_cls_size - train_enc_size

    # train_encoder_set, train_classifier_set, test_set = random_split(
    #     training_data, [train_enc_size, train_cls_size, test_size]
    # )

    return training_data, test_data

def get_datasets_seq(config):
    if config.dataset == "MNIST":
        training_data = torchvision.datasets.MNIST(
            train=True, download=True, root="./data",
            transform=torchvision.transforms.Compose(config.train_transform)
        )
        test_data = torchvision.datasets.MNIST(
            train=False, download=True, root="./data",
            transform=torchvision.transforms.Compose(config.test_transform)
        )
    elif config.dataset == "CIFAR10":
        training_data = torchvision.datasets.CIFAR10(
            train=True, download=True, root="./data",
            transform=torchvision.transforms.Compose(config.train_transform)
        )
        test_data = torchvision.datasets.CIFAR10(
            train=False, download=True, root="./data",
            transform=torchvision.transforms.Compose(config.test_transform)
        )
    elif config.dataset == "CIFAR100":
        training_data = torchvision.datasets.CIFAR100(
            train=True, download=True, root="./data",
            transform=torchvision.transforms.Compose(config.train_transform)
        )
        test_data = torchvision.datasets.CIFAR100(
            train=False, download=True, root="./data",
            transform=torchvision.transforms.Compose(config.test_transform)
        )
    elif config.dataset == "KMNIST":
        training_data = torchvision.datasets.KMNIST(
            train=True, download=True, root="./data",
            transform=torchvision.transforms.Compose(config.train_transform)
        )
        test_data = torchvision.datasets.KMNIST(
            train=False, download=True, root="./data",
            transform=torchvision.transforms.Compose(config.test_transform)
        )
    elif config.dataset == "FashionMNIST":
        training_data = torchvision.datasets.FashionMNIST(
            train=True, download=True, root="./data",
            transform=torchvision.transforms.Compose(config.train_transform)
        )
        test_data = torchvision.datasets.FashionMNIST(
            train=False, download=True, root="./data",
            transform=torchvision.transforms.Compose(config.test_transform)
        )
    else:
        print(f"Dataset '{config.dataset}' is not implemented.")
        raise NotImplementedError

    if config.limit_data < float('inf'):
        indices = torch.arange(config.limit_data)
        training_data = Subset(training_data, indices)

    training_data = DatasetWithIndices(training_data)

    total_size = len(training_data)
    train_enc_size = int(0.7 * total_size)
    train_cls_size = int(0.2 * total_size)
    test_size = total_size - train_cls_size - train_enc_size

    train_encoder_set, train_classifier_set, test_set = random_split(
        training_data, [train_enc_size, train_cls_size, test_size]
    )

    return train_encoder_set, train_classifier_set, test_set
