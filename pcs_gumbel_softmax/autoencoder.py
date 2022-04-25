import torch
from torch import nn

class Encoder_Stark(nn.Module):
    def __init__(self, in_features=2, width=2, out_features=2):
        super().__init__()
        self.lin1 = nn.Linear(in_features, width)
        # self.lin1.weight.data.fill_(1/width) # set weigths equal to uniform distribution
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(width, out_features)
        # self.lin2.weight.data.fill_(1 / width)
    def forward(self, y):
        y = self.act1(self.lin1(y))
        return self.lin2(y)

class Encoder_Aref(nn.Module):
    def __init__(self, in_features=2, width=2, out_features=2):
        super().__init__()
        self.lin1 = nn.Linear(in_features, width)
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(width, out_features)

    def forward(self, y):
        y = self.act1(self.lin1(y))
        return self.lin2(y)

class Decoder_Stark(nn.Module):
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


class Decoder_Aref(nn.Module):
    def __init__(self, in_features=2, width=2, out_features=2):
        super().__init__()
        self.lin1 = nn.Linear(in_features, width)
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(width, width)
        self.act2 = nn.ReLU()
        self.lin3 = nn.Linear(width, width)
        self.act3 = nn.ReLU()
        self.lin4 = nn.Linear(width, width)
        self.act4 = nn.ReLU()
        self.lin5 = nn.Linear(width, width)
        self.act5 = nn.ReLU()
        self.lin6 = nn.Linear(width, out_features)


    def forward(self, y):
        y = self.act1(self.lin1(y))
        y = self.act2(self.lin2(y))
        y = self.act3(self.lin3(y))
        y = self.act4(self.lin4(y))
        y = self.act5(self.lin5(y))
        return self.lin6(y)
