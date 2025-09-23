import torch
import torch.nn as nn

from sirentv.loss.builder import REGULARIZERS

@REGULARIZERS.register_module()
class LNRegularization(nn.Module):
    def __init__(self, weight_decay, p=2):
        super().__init__()
        self.weight_decay = weight_decay
        self.p = p

    def forward(self, net):
        return self.weight_decay * torch.norm(net.parameters(), p=self.p)


@REGULARIZERS.register_module()
class L2Regularization(LNRegularization):
    def __init__(self, weight_decay):
        super().__init__(weight_decay, p=2)


@REGULARIZERS.register_module()
class L1Regularization(LNRegularization):
    def __init__(self, weight_decay):
        super().__init__(weight_decay, p=1)