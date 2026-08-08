import torch
from torch import nn


class Swish(nn.Module):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        super().__init__()

    def forward(self, X: torch.Tensor):
        return X * torch.sigmoid(self.alpha * X)
