import torch
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        return A * torch.sigmoid(self.alpha * A) * B
