import torch
from torch import nn


class SGD(nn.Module):
    def __init__(self, params, lr, momentum, weight_decay, nesterov):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov

    def zero_grad(self, set_to_none: bool = False) -> None:
        pass

    def step(self):
        pass