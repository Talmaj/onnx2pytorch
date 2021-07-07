import torch
from torch import nn


class PRelu(nn.Module):
    def forward(self, X: torch.Tensor, slope: torch.Tensor):
        return torch.clamp(X, min=0) + torch.clamp(X, max=0) * slope
