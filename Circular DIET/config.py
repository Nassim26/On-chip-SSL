import torch
import numpy as np
import torchvision

class UnnormalizeTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return self.factor * x

class Config:
    def __init__(self):
        self.num_epoch = 10
        self.batch_size = 1024
        self.output_size = 512
        self.lr = 1e-3
        self.weight_decay = 0.05
        self.label_smoothing = 0.8
        self.num_classes = 10
        self.limit_data = np.inf  # Use `np.inf` for full dataset
        self.in_channels = 1
        self.dataset = "CIFAR10"
        self.input_shape = (3, 32, 32)
        self.test_transform = [
            torchvision.transforms.ToTensor(),
            #UnnormalizeTransform(255),
        ]
        self.train_transform = self._set_train_transform(augmentation_strenght=2)
        
    def _set_train_transform(self, augmentation_strenght):
        train_transform = self.test_transform
        if augmentation_strenght > 0:
            train_transform += [
                torchvision.transforms.RandomResizedCrop(self.input_shape[1], antialias=True),
                torchvision.transforms.RandomHorizontalFlip(),
            ]
        if augmentation_strenght > 1:
            train_transform += [
                torchvision.transforms.RandomApply(torch.nn.ModuleList(
                    [torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)]
                ), p=0.3),
                torchvision.transforms.RandomGrayscale(p=0.2)
            ]
        if augmentation_strenght > 2: 
            train_transform += [
                torchvision.transforms.RandomApply(torch.nn.ModuleList(
                    [torchvision.transforms.GaussianBlur((3, 3), (1.0, 2.0))]
                ), p=0.2),
                torchvision.transforms.RandomErasing(p=0.25)
            ]

        return train_transform