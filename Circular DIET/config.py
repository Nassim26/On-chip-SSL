import torch
import torchvision

class UnnormalizeTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return self.factor * x

class Config:
    def __init__(self):
        self.num_epoch = 25
        self.batch_size = 1024
        self.output_size = 10
        self.lr = 1e-3
        self.weight_decay = 0.05
        self.label_smoothing = 0.8
        self.num_classes = 10
        self.limit_data = 50 * self.batch_size  # Use `float('inf')` for full dataset
        self.in_channels = 1
        self.transform = [
            torchvision.transforms.ToTensor(),
            UnnormalizeTransform(255),
        ]