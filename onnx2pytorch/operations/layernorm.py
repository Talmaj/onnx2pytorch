from typing import Optional

import torch
from torch import nn

from onnx2pytorch.dtypes import ONNX_DTYPE_TO_TORCH


class LayerNorm(nn.Module):
    """
    ONNX LayerNormalization, which normalizes over the axes [axis, ..., rank-1].

    The mean and variance are accumulated in stash_type, float by default, and
    the normalized tensor is cast back to the input type before it is scaled.
    Mean and InvStdDev are returned in the accumulation type.
    """

    def __init__(self, eps: float = 1e-05, axis: int = -1, stash_type: int = 1):
        super().__init__()
        self.eps = eps
        self.axis = axis
        self.stash_type = stash_type
        self.accumulation_dtype = ONNX_DTYPE_TO_TORCH.get(stash_type)
        if (
            self.accumulation_dtype is None
            or not self.accumulation_dtype.is_floating_point
        ):
            raise NotImplementedError(
                "LayerNormalization stash_type {} is not supported in "
                "pytorch.".format(stash_type)
            )

    def forward(
        self,
        inputs: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ):
        axis = self.axis if self.axis >= 0 else inputs.dim() + self.axis
        dims = tuple(range(axis, inputs.dim()))

        values = inputs.to(self.accumulation_dtype)
        mean = values.mean(dim=dims, keepdim=True)
        deviation = values - mean
        variance = deviation.square().mean(dim=dims, keepdim=True)
        inv_std_dev = torch.rsqrt(variance + self.eps)

        normalized = (deviation * inv_std_dev).to(inputs.dtype)
        if scale is not None:
            normalized = normalized * scale
        if bias is not None:
            normalized = normalized + bias
        return normalized, mean, inv_std_dev

    def extra_repr(self) -> str:
        return "axis={}, eps={}, stash_type={}".format(
            self.axis, self.eps, self.stash_type
        )
