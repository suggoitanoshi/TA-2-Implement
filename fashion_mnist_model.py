import torch.nn as nn
import torch.nn.functional as F
from utils import num_classes


class FashionMnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.elu1 = nn.ELU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.elu2 = nn.ELU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=32*7*7, out_features=num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.elu1(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.elu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out
