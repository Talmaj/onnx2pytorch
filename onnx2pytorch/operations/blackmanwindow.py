import torch
from torch import nn

from onnx2pytorch.utils import cosine_window, get_torch_dtype


class BlackmanWindow(nn.Module):
    """ONNX BlackmanWindow: Blackman window of the given size."""

    def __init__(self, output_datatype=1, periodic=1):
        super().__init__()
        self.output_datatype = output_datatype
        self.periodic = periodic

    def forward(self, size: torch.Tensor):
        window = cosine_window(size, self.periodic, (0.42, -0.5, 0.08))
        return window.to(get_torch_dtype(self.output_datatype))
