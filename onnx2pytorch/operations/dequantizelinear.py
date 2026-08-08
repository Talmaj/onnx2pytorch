import torch
from torch import nn

from onnx2pytorch.operations.quantizelinear import broadcast_scale


class DequantizeLinear(nn.Module):
    """ONNX DequantizeLinear: y = (x - x_zero_point) * x_scale."""

    def __init__(self, dim=1, block_size=0):
        super().__init__()
        self.dim = dim
        self.block_size = block_size

    def forward(
        self,
        x: torch.Tensor,
        x_scale: torch.Tensor,
        x_zero_point: torch.Tensor = None,
    ):
        dim = self.dim if self.dim >= 0 else self.dim + x.ndim
        dtype = x_scale.dtype
        compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32

        y = x.to(compute_dtype)
        if x_zero_point is not None:
            y = y - broadcast_scale(
                x, x_zero_point.to(compute_dtype), dim, self.block_size
            )
        y = y * broadcast_scale(x, x_scale.to(compute_dtype), dim, self.block_size)
        return y.to(dtype)
