import torch
from torch import nn

from onnx2pytorch.dtypes import ONNX_DTYPE_TO_TORCH


def broadcast_scale(x, scale, dim, block_size):
    """Reshape a per-tensor, per-axis or blocked scale so that it broadcasts to x."""
    if scale.ndim == 0:
        return scale
    if scale.ndim == 1 and x.ndim > 1:
        shape = [1] * x.ndim
        shape[dim] = -1
        return scale.reshape(shape)
    if scale.ndim == x.ndim and block_size:
        scale = scale.repeat_interleave(block_size, dim=dim)
        return scale.narrow(dim, 0, x.shape[dim])
    return scale


class QuantizeLinear(nn.Module):
    """ONNX QuantizeLinear: y = saturate(round(x / y_scale) + y_zero_point)."""

    def __init__(self, dim=1, block_size=0, output_dtype=0, saturate=1):
        super().__init__()
        self.dim = dim
        self.block_size = block_size
        self.output_dtype = output_dtype
        self.saturate = saturate

    def forward(
        self,
        x: torch.Tensor,
        y_scale: torch.Tensor,
        y_zero_point: torch.Tensor = None,
    ):
        if y_zero_point is not None:
            dtype = y_zero_point.dtype
        elif self.output_dtype:
            dtype = ONNX_DTYPE_TO_TORCH.get(self.output_dtype)
            if dtype is None:
                raise NotImplementedError(
                    "QuantizeLinear with output_dtype={} not implemented.".format(
                        self.output_dtype
                    )
                )
        else:
            dtype = torch.uint8

        dim = self.dim if self.dim >= 0 else self.dim + x.ndim
        compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        scale = broadcast_scale(x, y_scale.to(compute_dtype), dim, self.block_size)
        y = torch.round(x.to(compute_dtype) / scale)
        if y_zero_point is not None:
            y = y + broadcast_scale(
                x, y_zero_point.to(compute_dtype), dim, self.block_size
            )

        if dtype.is_floating_point:
            if not self.saturate:
                return y.to(dtype)
            info = torch.finfo(dtype)
            return y.clamp(info.min, info.max).to(dtype)
        info = torch.iinfo(dtype)
        return y.clamp(info.min, info.max).to(dtype)
