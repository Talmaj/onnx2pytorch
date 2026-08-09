import torch
from torch import nn


class DFT(nn.Module):
    """ONNX DFT: discrete Fourier transform along one axis of a real or complex signal."""

    def __init__(self, opset_version, dim=None, inverse=0, onesided=0):
        super().__init__()
        # The axis moved from an attribute to an input in opset 20
        if dim is None:
            dim = -2 if opset_version >= 20 else 1
        self.dim = dim
        self.inverse = inverse
        self.onesided = onesided

    def forward(
        self,
        input: torch.Tensor,
        dft_length: torch.Tensor = None,
        axis: torch.Tensor = None,
    ):
        dim = self.dim if axis is None else int(axis)
        dim = dim % input.ndim
        n = input.shape[dim] if dft_length is None else int(dft_length)

        if input.shape[-1] == 1:
            signal = input[..., 0]
        else:
            signal = torch.complex(input[..., 0], input[..., 1])

        transform = torch.fft.ifft if self.inverse else torch.fft.fft
        result = transform(signal, n=n, dim=dim)
        output = torch.stack((result.real, result.imag), dim=-1)

        if self.onesided:
            output = output.narrow(dim, 0, output.shape[dim] // 2 + 1)
        return output.to(input.dtype)
