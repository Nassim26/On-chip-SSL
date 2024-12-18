import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import save_logs, save_model, get_datasets, get_datasets_seq

##############################
# Circular DIET training!    #
##############################

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

    try:
        embedding_dim = net.embedding_dim
    except:
        embedding_dim = net.fc.in_features

    W_probe = torch.nn.Linear(embedding_dim, config.num_classes).to(device)
    W_diet = torch.nn.Linear(embedding_dim, config.output_size, bias=False).to(device)

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

    save_model(net, "params.pth")

##############################
# Sequential DIET training!  #
##############################

def train_sequential(net, device, config):
    # Data loaders
    training_data_seq, probe_training_data, test_data_seq = get_datasets_seq(config)
    
    training_loader_diet = DataLoader(
        training_data_seq, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2
    )
    training_loader_probe = DataLoader(
        probe_training_data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2
    )
    test_loader = DataLoader(
        test_data_seq, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2
    )

    # Probes and optimizers
    embedding_dim = net.embedding_dim
    W_probe = torch.nn.Linear(embedding_dim, config.num_classes).to(device)
    W_diet = torch.nn.Linear(embedding_dim, config.output_size, bias=False).to(device)

    optimizer_diet = torch.optim.AdamW(
        list(net.parameters()) + list(W_diet.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    optimizer_probe = torch.optim.AdamW(
        list(W_probe.parameters()),
        lr=lr, weight_decay=weight_decay
    )

    criterion_diet = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    criterion_probe = torch.nn.CrossEntropyLoss(label_smoothing=0.0)

    # Train W_DIET
    print("Training W_DIET...")
    for epoch in range(num_epoch):
        net.train()
        run_loss_diet = []
        for x, _, n in tqdm(training_loader_diet, desc=f"Epoch {epoch + 1}/{num_epoch}"):
            x, n = x.to(device), (n % output_size).to(device).view(-1).long()
            z = net(x)
            logits_diet = W_diet(z)
            loss_diet = criterion_diet(logits_diet, n)

            optimizer_diet.zero_grad()
            loss_diet.backward()
            optimizer_diet.step()

            run_loss_diet.append(loss_diet.item())

        print(f"Epoch {epoch + 1}: Loss_DIET = {np.mean(run_loss_diet):.4f}")

    # Freeze W_DIET weights
    for param in W_diet.parameters():
        param.requires_grad = False

    # Train W_Probe
    print("\nTraining W_Probe...")
    for epoch in range(num_epoch):
        net.train()
        run_loss_probe = []
        run_acc = []
        for x, y, _ in tqdm(training_loader_probe, desc=f"Epoch {epoch + 1}/{num_epoch}"):
            x, y = x.to(device), y.to(device).long()
            z = net(x)
            logits_probe = W_probe(z)
            loss_probe = criterion_probe(logits_probe, y)

            optimizer_probe.zero_grad()
            loss_probe.backward()
            optimizer_probe.step()

            run_loss_probe.append(loss_probe.item())
            run_acc.append((logits_probe.argmax(1) == y).float().mean().item())

        print(
            f"Epoch {epoch + 1}: Loss_Probe = {np.mean(run_loss_probe):.4f}, "
            f"Accuracy = {np.mean(run_acc):.4f}"
        )

    # Test W_Probe
    print("\nTesting W_Probe...")
    net.eval()
    run_acc_test = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            z = net(x)
            logits_probe = W_probe(z)
            run_acc_test.append((logits_probe.argmax(1) == y).float().mean().item())

    print(f"Test Accuracy = {np.mean(run_acc_test):.4f}")

##############################
# Standard DIET training!    #
############################## 

def train_DIET_standard(net, device, config):
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
    W_diet = torch.nn.Linear(net.embedding_dim, len(training_data), bias=False).to(device)

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
            x, y, n = x.to(device), y.to(device), n.to(device).view(-1).long()
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