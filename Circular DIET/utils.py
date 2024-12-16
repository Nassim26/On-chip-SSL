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
    training_data = torchvision.datasets.MNIST(
        train=True, download=True, root="./data",
        transform=torchvision.transforms.Compose(config.transform)
    )

    if config.limit_data < float('inf'):
        indices = torch.arange(config.limit_data)
        training_data = Subset(training_data, indices)

    training_data = DatasetWithIndices(training_data)

    test_data = torchvision.datasets.MNIST(
        train=False, download=False, root="./data",
        transform=torchvision.transforms.Compose(config.transform)
    )

    return training_data, test_data