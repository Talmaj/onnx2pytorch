import torch
from torch import nn
from typing import Optional


class LayerNorm(nn.Module):  # pylint: disable=missing-docstring
    def __init__(self, normalized_shape: list, eps: float = 1e-05, axis: int = -1):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.axis = axis

    def forward(  # pylint: disable=missing-function-docstring
        self,
        inputs: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # ONNX LayerNormalization normalizes over [axis, ..., rank-1]
        # Convert negative axis to positive
        axis = self.axis if self.axis >= 0 else len(inputs.shape) + self.axis

        # Compute normalized_shape from input dimensions [axis, ..., rank-1]
        # This must match the scale/bias shape
        normalized_shape = list(inputs.shape[axis:])

        return nn.functional.layer_norm(
            input=inputs,
            normalized_shape=normalized_shape,
            weight=scale,
            bias=bias,
            eps=self.eps,
        )
