import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import save_logs, save_model, get_datasets, get_datasets_seq

##############################
# Random DIET               #
##############################

def train_random(net, device, config, embedding_dim=None):
    if embedding_dim == None:
        try: 
            embedding_dim = net.embedding_dim
        except: 
            print("Please provide the encoder output dimension.")

    training_data, test_data = get_datasets(config)

    training_loader = DataLoader(
        training_data, batch_size=config.batch_size,
        shuffle=True, drop_last=False, num_workers=2
    )
    test_loader = DataLoader(
        test_data, batch_size=config.batch_size,
        shuffle=False, drop_last=False, num_workers=2
    )

    W_probe = torch.nn.Linear(embedding_dim, config.num_classes).to(device)
    print("Embedding_dim size:", embedding_dim)
    W_diet = torch.nn.Linear(embedding_dim, embedding_dim, bias=False).to(device)

    optimizer = torch.optim.AdamW(
        list(net.parameters()) + list(W_probe.parameters()) + list(W_diet.parameters()),
        lr=config.lr, weight_decay=config.weight_decay
    )

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
    criterion_diet = torch.nn.CrossEntropyLoss(label_smoothing=0.0)

    for epoch in range(config.num_epoch):
        net.train()
        run_loss_diet, run_acc = [], []

        for x, y, n in tqdm(training_loader, desc=f"Epoch {epoch+1}/{config.num_epoch}"): 
            x, y = x.to(device), y.to(device)
            z = net(x)
            logits_diet = W_diet(z)
            n = n.view(-1).long()  # Ensure `n` is on the correct device and has the expected shape
            
            target = torch.stack([
                torch.rand(1, generator=torch.Generator().manual_seed(int(seed)))
                for seed in n
            ]).squeeze().to(device)

            loss_diet = criterion_diet(logits_diet, target)
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