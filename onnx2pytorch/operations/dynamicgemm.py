"""Gemm whose B (and optional C) are computed instead of stored as parameters."""

import torch
from torch import nn


class DynamicGemm(nn.Module):
    """ONNX Gemm: ``alpha * A @ B + beta * C``, with the usual transposes.

    ``transpose_weight`` follows the static Gemm converter: it is true when
    ONNX ``transB`` is 0, because ``nn.Linear`` transposes its parameter.
    """

    def __init__(
        self,
        transpose_activation=False,
        transpose_weight=True,
        weight_multiplier=1.0,
        bias_multiplier=1.0,
    ):
        super().__init__()
        self.transpose_activation = bool(transpose_activation)
        self.transpose_weight = bool(transpose_weight)
        self.weight_multiplier = float(weight_multiplier)
        self.bias_multiplier = float(bias_multiplier)

    def forward(self, a, b, c=None):
        if self.transpose_activation:
            a = a.transpose(-1, -2)
        # transpose_weight is the inverse of ONNX transB.
        if not self.transpose_weight:
            b = b.transpose(-1, -2)
        result = a.matmul(b) * self.weight_multiplier
        if c is not None:
            result = result + c * self.bias_multiplier
        return result
