import torch
from torch import nn


class Max(nn.Module):
    """ONNX Max, an elementwise maximum over 1 to N broadcast inputs."""

    def forward(self, *inputs):
        return torch.amax(torch.stack(torch.broadcast_tensors(*inputs)), dim=0)


class Min(nn.Module):
    """ONNX Min, an elementwise minimum over 1 to N broadcast inputs."""

    def forward(self, *inputs):
        return torch.amin(torch.stack(torch.broadcast_tensors(*inputs)), dim=0)
