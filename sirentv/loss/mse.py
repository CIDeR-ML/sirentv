import torch
import torch.nn as nn


class WeightedL2Loss(nn.Module):
    """
    A simple loss module that implements a weighted MSE loss
    """

    def __init__(self, reduce_method=torch.mean):
        super().__init__()
        self.reduce = reduce_method

    def forward(self, pred, target, weight=1.0):
        loss = weight * (pred - target) ** 2
        return self.reduce(loss)

class L2Loss(nn.Module):
    """
    A simple loss module that implements a regular MSE loss
    """

    def __init__(self, reduce_method=torch.mean):
        super().__init__()
        self.reduce = reduce_method

    def forward(self, pred, target, weight=1.0):
        loss = (pred - target) ** 2
        return self.reduce(loss)


class UncertainMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, weights, log_sigma):
        sigma = torch.exp(log_sigma)
        loss = torch.mean(weights * (((target - pred) ** 2) / (sigma**2) + log_sigma))
        return loss
