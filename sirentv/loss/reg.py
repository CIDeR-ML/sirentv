import torch
import torch.nn as nn

class LNRegularization(nn.Module):
    def __init__(self, weight_decay, p=2):
        super().__init__()
        self.weight_decay = weight_decay
        self.p = p

    def forward(self, net):
        return self.weight_decay * torch.norm(net.parameters(), p=self.p)


class L2Regularization(LNRegularization):
    def __init__(self, weight_decay):
        super().__init__(weight_decay, p=2)


class L1Regularization(LNRegularization):
    def __init__(self, weight_decay):
        super().__init__(weight_decay, p=1)