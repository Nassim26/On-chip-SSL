import Dataset
import KRIAInterface
import torchvision
import torch
from torchvision.transforms import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def relu255(x):
    return torch.clamp(x, 0, 255)

class ReLU255(nn.Module):
    def __init__(self):
        super(ReLU255, self).__init__()

    def forward(self, x):
        return relu255(x)


class SampleNetwork(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(SampleNetwork, self).__init__()
        self.hiddenLayer = KRIAInterface.Conv2D_3x3(inChannels, outChannels, bias=False)
        self.actFunc = ReLU255()

    def forward(self, x):
        x = self.hiddenLayer(x)
        x = self.actFunc(x)
        return x

dataset = Dataset.Dataset("MNIST", normalization=None, batchSize=64, classificationSplit=0.1)
_, (batch, label) = next(enumerate(dataset.trainingEnumeration()))

network = SampleNetwork(1, 1)
with torch.no_grad():
    network.hiddenLayer.weight = nn.Parameter(torch.tensor([[[[1,2,1], [2,4,2], [1,2,1]]]])/16) # We manually set the Gaussian blur kernel as an example

results = network(batch)

ax1 = plt.subplot(121)
ax1.imshow(batch[0][0].detach().numpy(), cmap="gray")
ax1.set_title("Input image")
ax2 = plt.subplot(122)
ax2.imshow(results[0][0].detach().numpy(), cmap="gray")
ax2.set_title("Output of CNN")
plt.show()