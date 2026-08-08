import torch
from torch import nn
from torch.nn import functional as F

from onnx2pytorch.operations.autopad import AutoPad


class ConvInteger(nn.Module):
    """ONNX ConvInteger: convolution of quantized inputs producing int32 output."""

    def __init__(
        self, kernel_size=None, dilation=1, padding=0, stride=1, groups=1, auto_pad=None
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.groups = groups

        self.padding = 0
        self.pad_layer = None
        if auto_pad is not None:
            self.pad_layer = AutoPad(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                mode=auto_pad,
            )
        elif isinstance(padding, nn.Module):
            self.pad_layer = padding
        else:
            self.padding = padding

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        x_zero_point: torch.Tensor = None,
        w_zero_point: torch.Tensor = None,
    ):
        x = x.to(torch.float64)
        w = w.to(torch.float64)
        if x_zero_point is not None:
            x = x - x_zero_point.to(torch.float64).reshape(())
        if w_zero_point is not None:
            zero_point = w_zero_point.to(torch.float64).flatten()
            w = w - zero_point.reshape([-1] + [1] * (w.ndim - 1))

        if self.pad_layer is not None:
            x = self.pad_layer(x)

        spatial = x.ndim - 2
        conv = getattr(F, "conv{}d".format(spatial))
        out = conv(
            x,
            w,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return torch.round(out).to(torch.int32)
