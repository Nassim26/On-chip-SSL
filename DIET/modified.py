from functools import partial
import numpy as np
from tqdm import tqdm

import torch
# from torchsummary import summary
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from resnetlike import resnet18_nores
from convnextlike import convnext_tiny_nores
from ournet import OurNet, ReLU255, LayerNorm
from datetime import datetime
from KRIAInterface import Conv2D_3x3

device = torch.device("cuda:2") # torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epoch = 500
batch_size = 1024
da_strength = 3
lr = 1e-3
weight_decay = 0.05
label_smoothing = 0.8
limit_data = batch_size * 5  # np.inf to train with whole training set

class DatasetWithIndices(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        # avoids class ordering in dataset:
        self.permutation = np.random.permutation(len(dataset))

    def __getitem__(self, n):
        x, y = self.dataset[n]
        n = torch.tensor([self.permutation[n]])
        return x, y, n

    def __len__(self):
        return len(self.dataset)


num_classes = 10
test_transform = [
    torchvision.transforms.ToTensor(),
]
if da_strength > 0:
  train_transform = test_transform + [
      torchvision.transforms.RandomResizedCrop(32, antialias=True),
      torchvision.transforms.RandomHorizontalFlip(),
  ]
if da_strength > 1:
    train_transform += [
        torchvision.transforms.RandomApply(torch.nn.ModuleList(
            [torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)]
        ), p=0.3),
        torchvision.transforms.RandomGrayscale(p=0.2)
    ]
if da_strength > 2:
    train_transform += [
        torchvision.transforms.RandomApply(torch.nn.ModuleList(
            [torchvision.transforms.GaussianBlur((3, 3), (1.0, 2.0))]
        ), p=0.2),
        torchvision.transforms.RandomErasing(p=0.25)
    ]

training_data = torchvision.datasets.CIFAR10(
    train=True, download=True, root='./data',
    transform=torchvision.transforms.Compose(train_transform)
)
if limit_data < np.inf:
  indices = torch.arange(limit_data)
  training_data = Subset(training_data, indices)
training_data = DatasetWithIndices(training_data)
test_data = torchvision.datasets.CIFAR10(
    train=False, download=True, root='./data',
    transform=torchvision.transforms.Compose(test_transform)
)

def train(net):
  # make loaders
  training_loader = DataLoader(
      training_data, batch_size=batch_size,
      shuffle=True, drop_last=False, num_workers=2
  )
  test_loader = DataLoader(
      test_data, batch_size=batch_size,
      shuffle=False, drop_last=False, num_workers=2
  )

  embedding_dim = net.embedding_dim
  W_probe = torch.nn.Linear(embedding_dim, num_classes).to(device)
  W_diet = torch.nn.Linear(embedding_dim, training_data.__len__(), bias=False).to(device)

  optimizer = torch.optim.AdamW(
      list(net.parameters()) + list(W_probe.parameters()) + list(W_diet.parameters()),
      lr=lr, weight_decay=weight_decay
  )

  criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
  criterion_diet = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

  with open(datetime.now().strftime("log_%Y_%m_%d_%H_%M_%S.txt"), "w") as file:
    pbar = tqdm(np.arange(num_epoch))
    for epoch in pbar:
        # Train
        net.train()
        run_loss_diet, run_acc = [], []
        for i, (x, y, n) in enumerate(training_loader):
          x = x.to(device)
          y = y.to(device)
          n = n.to(device).view(-1)
          z = net(x)
          logits_diet = W_diet(z)
          loss_diet = criterion_diet(logits_diet, n)
          logits_probe = W_probe(z.detach())
          loss_probe = criterion(logits_probe, y)
          loss = loss_diet + loss_probe
          optimizer.zero_grad()
          #loss.backward()
          loss_diet.backward()
          loss_probe.backward()
          optimizer.step()
          run_loss_diet.append(loss_diet.item())
          run_acc.append(torch.mean((y == logits_probe.argmax(1)).to(float)).item())

          # logging
          pbar.set_description('epoch: %s/%s, iter: %s/%s, loss_diet=%.4e, accuracy=%.4f' % (
              epoch, num_epoch, i, len(training_loader),
              np.mean(run_loss_diet), np.mean(run_acc)))
        file.write(f"{epoch},{np.mean(run_loss_diet)},{np.mean(run_acc)}\n")
        file.flush()
    print('\nTraining done.')

    # Test
    net.eval()
    with torch.no_grad():
      run_acc_test = []
      for j, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        z = net(x)
        logits_probe = W_probe(z.detach())
        run_acc_test.append(torch.mean((y == logits_probe.argmax(1)).to(float)).item())
    print('Test accuracy=%.4f' % np.mean(run_acc_test))
    file.write(f"Final,,{np.mean(run_acc_test)}\n")
    file.flush()



settings = [
  {#custom
    "channels": [3, 64, 128, 256],
    "set_sizes": [0, 1, 1, 1],
    "pools": [0, 0, 3, 2]
  },
  # {#baseline
  #   "channels": [3, 64, 128, 256, 512],
  #   "set_sizes": [0, 5, 4, 4, 4],
  #   "pools": [0, 0, 2, 2, 2]
  # },
  # {#less end channels
  #   "channels": [3, 64, 128, 256, 256],
  #   "set_sizes": [0, 5, 4, 4, 4],
  #   "pools": [0, 0, 2, 2, 2]
  # },
  # {#less intermediate channels
  #   "channels": [3, 64, 128, 128, 512],
  #   "set_sizes": [0, 5, 4, 4, 4],
  #   "pools": [0, 0, 2, 2, 2]
  # },
  # {#less deep
  #   "channels": [3, 64, 128, 256, 512],
  #   "set_sizes": [0, 4, 3, 3, 3],
  #   "pools": [0, 0, 2, 2, 2]
  # },
  # {#even less deep
  #   "channels": [3, 64, 128, 256, 512],
  #   "set_sizes": [0, 3, 2, 2, 2],
  #   "pools": [0, 0, 2, 2, 2]
  # },
  # {#faster pooling
  #   "channels": [3, 64, 64, 128, 256, 512],
  #   "set_sizes": [0, 1, 4, 4, 4, 4],
  #   "pools": [0, 0, 2, 0, 2, 2]
  # }
]

layer_types = [
  {# custom
    "conv3x3_layer": partial(torch.nn.Conv2d, kernel_size=3, stride=1, padding=0, bias=False),
    "norm_layer": LayerNorm,
    "relu_layer": torch.nn.ReLU,
  },
  # {# resnet18like
  #   "conv3x3_layer": partial(torch.nn.Conv2d, kernel_size=3, stride=1, padding=1, bias=False),
  #   "norm_layer": torch.nn.BatchNorm2d,
  #   "relu_layer": torch.nn.ReLU,
  # },
  # {# no norm, with bias
  #   "conv3x3_layer": partial(torch.nn.Conv2d, kernel_size=3, stride=1, padding=1, bias=True),
  #   "norm_layer": torch.nn.Identity,
  #   "relu_layer": torch.nn.ReLU,
  # },
  # {# quant convolution + relu255
  #   "conv3x3_layer": partial(Conv2D_3x3, bias=False),
  #   "norm_layer": torch.nn.BatchNorm2d,
  #   "relu_layer": ReLU255,
  # },
  # {# full quant, no norm, with bias
  #   "conv3x3_layer": partial(Conv2D_3x3, bias=True),
  #   "norm_layer": torch.nn.Identity,
  #   "relu_layer": ReLU255,
  # },
]

for l in layer_types:
  for s in settings:
    # net = resnet18_nores()

    # # modify for small (32x32) Dataset
    # net.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    # net.maxpool = torch.nn.Identity()
    # embedding_dim = net.fc.in_features
    # net.fc = torch.nn.Identity()

    # net = convnext_tiny_nores()
    # embedding_dim = net.classifier[2].in_features
    # net.classifier[2] = torch.nn.Identity()
    
    net = OurNet(s["channels"], s["set_sizes"], s["pools"], l["conv3x3_layer"], l["norm_layer"], l["relu_layer"])
    embedding_dim = net.embedding_dim

    net.to(device)
    #print(summary(net, (3, 32, 32)))
    train(net)