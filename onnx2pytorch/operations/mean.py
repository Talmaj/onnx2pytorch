import torch
from torch import nn


class Mean(nn.Module):
    def forward(self, *inputs):
        return torch.mean(torch.stack(torch.broadcast_tensors(*inputs)), dim=0)
