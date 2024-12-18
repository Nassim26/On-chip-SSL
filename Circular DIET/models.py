import torch
import torch.nn as nn
from layers import ReLU255, QConv2D

### -== Networks ==- ###

class QuantizedNetwork(nn.Module):
    def __init__(self, in_channels, input_shape=(1, 28, 28)):
        super(QuantizedNetwork, self).__init__()
        self.hiddenLayer0 = QConv2D(in_channels, 16, bias=True)
        self.hiddenLayer1 = QConv2D(16, 32, bias=True)
        self.hiddenLayer2 = QConv2D(32, 64, bias=True)
        self.hiddenLayer3 = QConv2D(64, 128, bias=True)
        self.actFunc = ReLU255()
        self.maxPool2 = nn.MaxPool2d((2, 2), stride=2)
        self.Flatten = nn.Flatten()
        self.embedding_dim = self._calculate_embedding_dim(input_shape)

    def _calculate_embedding_dim(self, input_shape):
        dummy = torch.zeros(1, *input_shape)
        with torch.no_grad():
            y = self.forward(dummy)
        return y.shape[1]

    def forward(self, x):
        x = self.hiddenLayer0(x)
        x = self.actFunc(x)
        x = self.hiddenLayer1(x)
        x = self.actFunc(x)
        x = self.maxPool2(x)
        x = self.hiddenLayer2(x)
        x = self.actFunc(x)
        x = self.Flatten(x)
        return x

class LargeNetwork(nn.Module):
    def __init__(self, in_channels, input_shape):
        super(LargeNetwork, self).__init__()

class StandardNetwork(nn.Module):
    def __init__(self, in_channels, input_shape=(1,28,28)):
        super(StandardNetwork, self).__init__()
        self.Layers = [
            nn.Conv2d(in_channels, 16, kernel_size=3, strid=1, padding=(2,2), bias=True),
            nn.ReLU()
        ]
        