import torch
from torch import nn

from onnx2pytorch.dtypes import ONNX_DTYPE_TO_TORCH


class EyeLike(nn.Module):
    def __init__(self, dtype=None, k=0):
        super().__init__()
        self.dtype = dtype
        self.k = k

    def forward(self, input: torch.Tensor):
        if self.dtype is None:
            dtype = input.dtype
        else:
            dtype = ONNX_DTYPE_TO_TORCH.get(self.dtype)
            if dtype is None:
                raise ValueError(
                    "EyeLike dtype {} is not supported in pytorch.".format(self.dtype)
                )
        rows, cols = input.shape
        row_idx = torch.arange(rows, device=input.device).unsqueeze(1)
        col_idx = torch.arange(cols, device=input.device).unsqueeze(0)
        return (col_idx - row_idx == self.k).to(dtype)

    def extra_repr(self) -> str:
        return "dtype={}, k={}".format(self.dtype, self.k)
