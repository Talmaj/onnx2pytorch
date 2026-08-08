import torch
from torch import nn


class Sum(nn.Module):
    def forward(self, *inputs):
        return torch.sum(torch.stack(torch.broadcast_tensors(*inputs)), dim=0)
