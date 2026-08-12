import torch
from torch import nn
from torch.nn import functional as F

from onnx2pytorch.operations.autopad import AutoPad


class QLinearConv(nn.Module):
    """ONNX QLinearConv: convolution of quantized inputs with quantized output."""

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
        self.auto_pad = auto_pad
        if auto_pad is not None:
            # kernel_shape is optional, without it the pad follows from W
            if kernel_size is not None:
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
        x_scale: torch.Tensor,
        x_zero_point: torch.Tensor,
        w: torch.Tensor,
        w_scale: torch.Tensor,
        w_zero_point: torch.Tensor,
        y_scale: torch.Tensor,
        y_zero_point: torch.Tensor,
        b: torch.Tensor = None,
    ):
        dtype = y_zero_point.dtype
        x = x.to(torch.float64) - x_zero_point.to(torch.float64).reshape(())
        w = w.to(torch.float64) - w_zero_point.to(torch.float64).flatten().reshape(
            [-1] + [1] * (w.ndim - 1)
        )

        if self.pad_layer is None and self.auto_pad is not None:
            self.pad_layer = AutoPad(
                kernel_size=tuple(w.shape[2:]),
                stride=self.stride,
                dilation=self.dilation,
                mode=self.auto_pad,
            )
        if self.pad_layer is not None:
            x = self.pad_layer(x)

        spatial = x.ndim - 2
        conv = getattr(F, "conv{}d".format(spatial))
        acc = conv(
            x,
            w,
            None if b is None else b.to(torch.float64),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        channel_shape = [-1] + [1] * spatial
        scale = x_scale.to(torch.float64) * w_scale.to(torch.float64).flatten().reshape(
            channel_shape
        )
        y = torch.round(torch.round(acc) * scale / y_scale.to(torch.float64))
        y = y + y_zero_point.to(torch.float64).reshape(())

        info = torch.iinfo(dtype)
        return y.clamp(info.min, info.max).to(dtype)
