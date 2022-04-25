import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, in_features=2, width=2, out_features=2):
        super().__init__()
        self.lin1 = nn.Linear(in_features, width)
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(width, out_features)

    def forward(self, y):
        y = self.act1(self.lin1(y))
        return self.lin2(y)


class Decoder(nn.Module):
    def __init__(self, in_features=2, width=2, out_features=2):
        super().__init__()
        self.lin1 = nn.Linear(in_features, width)
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(width, width)
        self.act2 = nn.ReLU()
        self.lin3 = nn.Linear(width, out_features)


    def forward(self, y):
        y = self.act1(self.lin1(y))
        y = self.act2(self.lin2(y))
        return self.lin3(y)
