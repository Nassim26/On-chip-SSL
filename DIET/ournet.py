import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

def relu255(x):
    return torch.clamp(x, 0, 255)

class ReLU255(nn.Module):
    def __init__(self):
        super(ReLU255, self).__init__()

    def forward(self, x):
        return relu255(x)
    
# From https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py#L15
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class ConvLayer(nn.Module):
    def __init__(
        self,
        inplanes: int,
        outplanes: int,
        conv3x3_layer,
        norm_layer,
        relu_layer,
    ) -> None:
        super().__init__()
        self.conv = conv3x3_layer(inplanes, outplanes)
        self.norm = norm_layer(outplanes)
        self.relu = relu_layer()

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out
    
class LayerSet(nn.Module):
    def __init__(
        self,
        inplanes: int,
        outplanes: int,
        count: int,
        conv3x3_layer,
        norm_layer,
        relu_layer,
    ) -> None:
        super().__init__()
        layers = [ConvLayer(inplanes, outplanes, conv3x3_layer, norm_layer, relu_layer)]
        for _ in range(count - 1):
            layers.append(ConvLayer(outplanes, outplanes, conv3x3_layer, norm_layer, relu_layer))
        self.seq = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.seq(x)
        return out

class OurNet(nn.Module):
    def __init__(self, channels, set_sizes, pools, conv3x3_layer, norm_layer, relu_layer) -> None:
        super().__init__()
        layers = []
        for i in range(1, len(channels)):
            if pools[i] > 0:
                layers.append(nn.MaxPool2d(kernel_size=pools[i]))
            layers.append(LayerSet(channels[i - 1], channels[i], set_sizes[i], conv3x3_layer, norm_layer, relu_layer))

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.seq = nn.Sequential(*layers)
        self.embedding_dim = channels[-1]

    def forward(self, x: Tensor) -> Tensor:
        out = self.seq(x)
        out = torch.flatten(out, 1)
        return out
    
class SimpleNet(nn.Module):
    def __init__(self, channels, pools, conv3x3_layer, norm_layer, relu_layer) -> None:
        super().__init__()
        layers = []
        for i in range(1, len(channels)):
            if pools[i] > 0:
                layers.append(nn.MaxPool2d(kernel_size=pools[i]))
            layers.append(LayerSet(channels[i - 1], channels[i], conv3x3_layer, norm_layer, relu_layer))

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.seq = nn.Sequential(*layers)
        self.embedding_dim = channels[-1]

    def forward(self, x: Tensor) -> Tensor:
        out = self.seq(x)
        out = torch.flatten(out, 1)
        return out