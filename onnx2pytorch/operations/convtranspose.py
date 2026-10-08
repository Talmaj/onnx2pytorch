import torch
from torch import nn

from onnx2pytorch.utils import as_tuple


def crop_conv_transpose(
    output,
    input_size,
    stride,
    kernel_size,
    dilation,
    output_padding,
    pads,
    output_shape,
    auto_pad,
):
    """Crop a conv-transpose result the way ONNX pads/output_shape/auto_pad ask."""
    ndim = len(input_size)
    auto_pad = auto_pad or "NOTSET"
    if output_shape is None and auto_pad in ("NOTSET", "VALID"):
        resolved = pads or (0,) * (2 * ndim)
    else:
        stride = as_tuple(stride, ndim)
        kernel_size = as_tuple(kernel_size, ndim)
        dilation = as_tuple(dilation, ndim)
        output_padding = as_tuple(output_padding, ndim)
        wanted = output_shape
        if wanted is None:
            wanted = [input_size[i] * stride[i] for i in range(ndim)]
        elif len(wanted) > ndim:
            wanted = wanted[-ndim:]
        starts, ends = [], []
        for i in range(ndim):
            total = (
                stride[i] * (input_size[i] - 1)
                + output_padding[i]
                + (kernel_size[i] - 1) * dilation[i]
                + 1
                - wanted[i]
            )
            if auto_pad == "SAME_UPPER":
                starts.append(total // 2)
                ends.append(total - total // 2)
            else:
                starts.append(total - total // 2)
                ends.append(total // 2)
        resolved = tuple(starts) + tuple(ends)

    crop = []
    half = len(resolved) // 2
    for i in range(half - 1, -1, -1):
        crop.extend([-resolved[i], -resolved[half + i]])
    if any(crop):
        output = torch.nn.functional.pad(output, crop)
    return output


class ConvTranspose(nn.Module):
    """Applies the ONNX ConvTranspose padding, which crops the output.

    When output_shape or auto_pad is set the pads are derived from the wanted
    output size, otherwise the explicit pads are used as they are.
    """

    def __init__(self, conv, pads=None, output_shape=None, auto_pad="NOTSET"):
        super().__init__()
        self.conv = conv
        self.pads = tuple(pads) if pads else None
        self.output_shape = tuple(output_shape) if output_shape else None
        self.auto_pad = auto_pad

    def forward(self, input):
        output = self.conv(input)
        conv = self.conv
        return crop_conv_transpose(
            output,
            input.shape[2:],
            conv.stride,
            conv.kernel_size,
            conv.dilation,
            conv.output_padding,
            self.pads,
            self.output_shape,
            self.auto_pad,
        )
