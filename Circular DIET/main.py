from config import Config
from models import QuantizedNetwork
from train import train, train_DIET_standard
import torch
import torchvision

if __name__ == "__main__":
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    config = Config()
    #net = QuantizedNetwork(config.in_channels, config.input_shape).to(device)
    # --- 
    net = torchvision.models.resnet18()
    net.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    net.maxpool = torch.nn.Identity()
    embedding_dim = net.fc.in_features
    net.fc = torch.nn.Identity()
    # ---
    #state_dict = torch.load('cifar_params.pth')
    #net.load_state_dict(state_dict)
    net.to(device)
    train_simcirc(net, device, config, embedding_dim)
    # train(net, device, config, net.embedding_dim)
    # train(net, device, config)
    # train_DIET_standard(net, device, config)
    # train_DIET_standard(net, device, config)
    # train_DIET_standard(net, device, config)
