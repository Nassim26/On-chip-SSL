from config import Config
from models import QuantizedNetwork
from train import train
import torch

if __name__ == "__main__":
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    config = Config()
    net = QuantizedNetwork(config.in_channels).to(device)
    train(net, device, config)
