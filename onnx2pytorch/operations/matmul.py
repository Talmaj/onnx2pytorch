import torch
from torch import nn


class MatMul(nn.Module):
    def forward(self, A, V):
        return torch.matmul(A, V)
