import Dataset
import KRIAInterface
import KRIAInterface_clipped
import torchvision
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def relu255(x):
    return torch.clamp(x, 0, 255)

class ReLU255(nn.Module):
    def __init__(self):
        super(ReLU255, self).__init__()

    def forward(self, x):
        return relu255(x)

class UnnormalizeTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return self.factor * x

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

num_epoch = 15
batch_size = 1024
output_size = 10384
lr = 1e-3
weight_decay = 0.05
label_smoothing = 0.8
num_classes = 10
limit_data = np.inf  # np.inf to train with whole training set

transform = [
    torchvision.transforms.ToTensor(),
    UnnormalizeTransform(255)
]

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

training_data = torchvision.datasets.MNIST(
    train=True, download=True, root="\data",
    transform=torchvision.transforms.Compose(transform)
)

if limit_data < np.inf:
  indices = torch.arange(limit_data)
  training_data = Subset(training_data, indices)

training_data = DatasetWithIndices(training_data)
test_data = torchvision.datasets.MNIST(
    train=False, download=False, root="\data",
    transform=torchvision.transforms.Compose(transform)
)

epoch_losses, epoch_accuracies = [], []

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
  W_diet = torch.nn.Linear(embedding_dim, output_size, bias=False).to(device)

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
          y = y.to(device).long()
          n = (n % output_size).to(device).view(-1).long()
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
        
        epoch_losses.append(np.mean(run_loss_diet))
        epoch_accuracies.append(np.mean(run_acc))
        file.write(f"{epoch},{np.mean(run_loss_diet)},{np.mean(run_acc)}\n")
        file.flush()
    print('\nTraining done.')

    # Test
    net.eval()
    all_preds, all_labels = [], [] 
    with torch.no_grad():
      run_acc_test = []
      for j, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        z = net(x)
        logits_probe = W_probe(z.detach())
        all_preds.append(logits_probe.argmax(1).cpu().numpy())
        all_labels.append(y.cpu().numpy())
        run_acc_test.append(torch.mean((y == logits_probe.argmax(1)).to(float)).item())
    print('Test accuracy=%.4f' % np.mean(run_acc_test))
    file.write(f"Final,,{np.mean(run_acc_test)}\n")
    file.flush()

    # Save Model
    torch.save(net.state_dict(), "minimal_network_params.pth")

    # Generate and Save Figures
    # 1. Loss and Accuracy Plot
    plt.figure(figsize=(10,6))
    plt.plot(np.arange(num_epoch), epoch_losses, label='Loss', color='red', linewidth=2)
    plt.plot(np.arange(num_epoch), epoch_accuracies, label='Accuracy', color='blue', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('Training Loss and Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_accuracy.png', dpi=300)
    plt.show()

    # 2. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()

    # 3. Classification Report
    report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(num_classes)])
    print(report)
    with open('classification_report.txt', 'w') as f:
        f.write(report)

class MinimalNetwork(nn.Module):
    def __init__(self, inChannels):
        super(MinimalNetwork, self).__init__()
        self.hiddenLayer0 = KRIAInterface.Conv2D_3x3(inChannels, 16, bias=True)
        self.actFunc = ReLU255()
        self.maxPool1 = nn.MaxPool2d((2, 2), stride = 1)
        self.maxPool2 = nn.MaxPool2d((2, 2), stride = 2)
        self.hiddenLayer1 = KRIAInterface.Conv2D_3x3(16, 32, bias=True)
        self.Flatten = nn.Flatten()
        self.hiddenLayer2 = KRIAInterface.Conv2D_3x3(32, 64, bias=True)
        self.hiddenLayer3 = KRIAInterface.Conv2D_3x3(64, 128, bias=True)
        self.embedding_dim = 6400

    def forward(self, x):
        x = self.hiddenLayer0(x)
        x = self.actFunc(x)
        # x = self.maxPool2(x)
        x = self.hiddenLayer1(x)
        x = self.actFunc(x)
        x = self.maxPool2(x)
        x = self.hiddenLayer2(x)
        x = self.actFunc(x)
        # x = self.maxPool1(x)
        #x = self.hiddenLayer3(x)
        #x = self.actFunc(x)
        # x = self.maxPool1(x)
        x = self.Flatten(x)
        return x

train(MinimalNetwork(1).to(device))
