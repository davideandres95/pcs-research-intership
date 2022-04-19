import torch
from torch import nn


class DistGenerator(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.lin1 = nn.Linear(M, M)
        self.lin1.weight = nn.Parameter(torch.full((M, M), 1 / M))  # set weigths equal to uniform distribution

    def forward(self, y):
        return self.lin1(y)


class Mapper(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.lin1 = nn.Linear(M, 2)

    def forward(self, y):
        return self.lin1(y)


class Demapper(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.lin1 = nn.Linear(2, 2 * M)
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(2 * M, 2 * M)
        self.act2 = nn.ReLU()
        self.lin3 = nn.Linear(2 * M, 2 * M)
        self.act3 = nn.ReLU()
        self.lin4 = nn.Linear(2 * M, 2 * M)
        self.act4 = nn.ReLU()
        self.lin5 = nn.Linear(2 * M, M)

    def forward(self, y):
        y = self.act1(self.lin1(y))
        y = self.act2(self.lin2(y))
        y = self.act3(self.lin3(y))
        return self.lin5(self.act4(self.lin4(y)))
