import torch
from torch import nn


class MatMulInteger(nn.Module):
    """ONNX MatMulInteger: matrix product of quantized inputs with int32 output."""

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        a_zero_point: torch.Tensor = None,
        b_zero_point: torch.Tensor = None,
    ):
        a = a.to(torch.int32)
        b = b.to(torch.int32)
        if a_zero_point is not None:
            a_zero_point = a_zero_point.to(torch.int32)
            if a_zero_point.ndim == 1 and a_zero_point.numel() > 1:
                a_zero_point = a_zero_point.unsqueeze(-1)
            a = a - a_zero_point
        if b_zero_point is not None:
            b = b - b_zero_point.to(torch.int32)
        return torch.matmul(a, b)
