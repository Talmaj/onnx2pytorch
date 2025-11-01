import torch
from torch import nn
from typing import Optional


class LayerNorm(nn.Module):  # pylint: disable=missing-docstring
    def __init__(self, normalized_shape: list, eps: float):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(  # pylint: disable=missing-function-docstring
        self,
        inputs: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return nn.functional.layer_norm(
            input=inputs,
            normalized_shape=self.normalized_shape,
            weight=scale,
            bias=bias,
            eps=self.eps,
        )