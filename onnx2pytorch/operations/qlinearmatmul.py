import torch
from torch import nn


def per_row(value):
    """Reshape a per-row scale or zero point so that it broadcasts over the rows."""
    if value.ndim == 1 and value.numel() > 1:
        return value.unsqueeze(-1)
    return value


class QLinearMatMul(nn.Module):
    """ONNX QLinearMatMul: matrix product of quantized inputs with quantized output."""

    def forward(
        self,
        a: torch.Tensor,
        a_scale: torch.Tensor,
        a_zero_point: torch.Tensor,
        b: torch.Tensor,
        b_scale: torch.Tensor,
        b_zero_point: torch.Tensor,
        y_scale: torch.Tensor,
        y_zero_point: torch.Tensor,
    ):
        dtype = y_zero_point.dtype
        # Accumulate exactly in int32, as the runtimes do, and scale afterwards
        acc = torch.matmul(
            a.to(torch.int32) - per_row(a_zero_point.to(torch.int32)),
            b.to(torch.int32) - b_zero_point.to(torch.int32),
        )
        scale = per_row(a_scale.to(torch.float64)) * b_scale.to(torch.float64)
        y = acc.to(torch.float64) * scale / per_row(y_scale.to(torch.float64))
        y = torch.round(y) + per_row(y_zero_point.to(torch.float64))

        info = torch.iinfo(dtype)
        return y.clamp(info.min, info.max).to(dtype)
