import torch
from torch import nn


class CastLike(nn.Module):
    """ONNX CastLike: cast the input to the data type of the target tensor."""

    def forward(self, input: torch.Tensor, target_type: torch.Tensor):
        return input.to(target_type.dtype)
