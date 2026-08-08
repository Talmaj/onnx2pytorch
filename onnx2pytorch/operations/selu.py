import torch
from torch import nn


class Selu(nn.Module):
    def __init__(
        self, alpha=1.67326319217681884765625, gamma=1.05070102214813232421875
    ):
        self.alpha = alpha
        self.gamma = gamma
        super().__init__()

    def forward(self, X: torch.Tensor):
        return self.gamma * (
            torch.clamp(X, min=0) + torch.clamp(self.alpha * torch.expm1(X), max=0)
        )
