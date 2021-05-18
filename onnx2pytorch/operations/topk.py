import torch
from torch import nn


class TopK(nn.Module):
    def __init__(self, axis=-1, largest=1, sorted=1):
        self.axis = axis
        self.largest = largest
        self.sorted = sorted

    def forward(self, X: torch.Tensor, K: torch.Tensor):
        return torch.topk(X, K, dim=self.axis, largest=self.largest, sorted=sorted)
