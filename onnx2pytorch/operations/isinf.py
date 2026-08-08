import torch
from torch import nn


class IsInf(nn.Module):
    def __init__(self, detect_negative=1, detect_positive=1):
        self.detect_negative = bool(detect_negative)
        self.detect_positive = bool(detect_positive)
        super().__init__()

    def forward(self, X: torch.Tensor):
        if self.detect_negative and self.detect_positive:
            return torch.isinf(X)
        elif self.detect_negative:
            return torch.isneginf(X)
        elif self.detect_positive:
            return torch.isposinf(X)
        return torch.zeros_like(X, dtype=torch.bool)
