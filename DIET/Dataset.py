import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as F

class ToHalfTensor(torch.nn.Module):
    def forward(self, img):
        return F.to_tensor(img).half()

class UnnormalizeTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return self.factor * x

class Dataset:

    def __init__(self, datasetName, batchSize, normalization, classificationSplit):

        transform = transforms.Compose([
            #ToHalfTensor() if useHalfPrecision else transforms.ToTensor(),
            transforms.ToTensor(),
            UnnormalizeTransform(255)
            # transforms.Normalize(normalization[0], normalization[1])
        ])

        if (datasetName == 'EMNIST'):
            self.train = getattr(datasets, datasetName)('datasets', train=True, download=True,split='balanced', transform=transform)
            self.test = getattr(datasets, datasetName)('datasets', train=False, split='balanced', transform=transform)
        else:
            self.train = getattr(datasets, datasetName)('datasets', train=True, download=True, transform=transform)
            self.test = getattr(datasets, datasetName)('datasets', train=False, transform=transform)

        self.train, self.classification = torch.utils.data.random_split(self.train, [1 - classificationSplit, classificationSplit])

        self.trainLoader = torch.utils.data.DataLoader(self.train, batch_size=batchSize)
        self.classificationLoader = torch.utils.data.DataLoader(self.classification, batch_size=batchSize)
        self.testLoader = torch.utils.data.DataLoader(self.test, batch_size=batchSize)

        self.batchSize = batchSize

    def trainingEnumeration(self):
        return self.trainLoader

    def classificationEnumeration(self):
        return self.classificationLoader

    def testingEnumeration(self):
        return self.testLoader

    def trainBatchCount(self):
        return len(self.trainLoader.dataset)

    def classificationBatchCount(self):
        return len(self.classificationLoader.dataset)

    def testBatchCount(self):
        return len(self.testLoader.dataset)