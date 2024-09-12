import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epoch = 30
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
# make loaders
training_loader = DataLoader(
    training_data, batch_size=batch_size,
    shuffle=True, drop_last=False, num_workers=2
)
test_loader = DataLoader(
    test_data, batch_size=batch_size,
    shuffle=False, drop_last=False, num_workers=2
)

net = torchvision.models.resnet18()

# modify for small (32x32) Dataset
net.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
net.maxpool = torch.nn.Identity()
embedding_dim = net.fc.in_features
net.fc = torch.nn.Identity()
net.to(device)

W_probe = torch.nn.Linear(embedding_dim, num_classes).to(device)
W_diet = torch.nn.Linear(embedding_dim, training_data.__len__(), bias=False).to(device)

optimizer = torch.optim.AdamW(
    list(net.parameters()) + list(W_probe.parameters()) + list(W_diet.parameters()),
    lr=lr, weight_decay=weight_decay
)

criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
criterion_diet = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

print(net)

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
      loss.backward()
      optimizer.step()
      run_loss_diet.append(loss_diet.item())
      run_acc.append(torch.mean((y == logits_probe.argmax(1)).to(float)).item())

      # logging
      pbar.set_description('epoch: %s/%s, iter: %s/%s, loss_diet=%.4e, accuracy=%.4f' % (
          epoch, num_epoch, i, len(training_loader),
          np.mean(run_loss_diet), np.mean(run_acc)))
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