import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import save_logs, save_model, get_datasets

def train(net, device, config):
    training_data, test_data = get_datasets(config)

    training_loader = DataLoader(
        training_data, batch_size=config.batch_size,
        shuffle=True, drop_last=False, num_workers=2
    )
    test_loader = DataLoader(
        test_data, batch_size=config.batch_size,
        shuffle=False, drop_last=False, num_workers=2
    )

    W_probe = torch.nn.Linear(net.embedding_dim, config.num_classes).to(device)
    W_diet = torch.nn.Linear(net.embedding_dim, config.output_size, bias=False).to(device)

    optimizer = torch.optim.AdamW(
        list(net.parameters()) + list(W_probe.parameters()) + list(W_diet.parameters()),
        lr=config.lr, weight_decay=config.weight_decay
    )

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
    criterion_diet = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    for epoch in range(config.num_epoch):
        net.train()
        run_loss_diet, run_acc = [], []

        for x, y, n in tqdm(training_loader, desc=f"Epoch {epoch+1}/{config.num_epoch}"):
            x, y, n = x.to(device), y.to(device), (n % config.output_size).to(device).view(-1).long()
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
            run_acc.append(torch.mean((y == logits_probe.argmax(1)).float()).item())

        print(f"Epoch {epoch+1}: Loss={np.mean(run_loss_diet):.4f}, Accuracy={np.mean(run_acc):.4f}")

    save_model(net, "params.pth")