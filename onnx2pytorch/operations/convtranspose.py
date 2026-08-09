import torch
from torch import nn


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

    def _pads(self, input_size):
        ndim = len(input_size)
        if self.output_shape is None and self.auto_pad in ("NOTSET", "VALID"):
            return self.pads or (0,) * (2 * ndim)

        conv = self.conv
        output_shape = self.output_shape
        if output_shape is None:
            output_shape = [input_size[i] * conv.stride[i] for i in range(ndim)]
        elif len(output_shape) > ndim:
            # The shape may be given for the whole tensor or for the spatial axes
            output_shape = output_shape[-ndim:]

        starts, ends = [], []
        for i in range(ndim):
            total = (
                conv.stride[i] * (input_size[i] - 1)
                + conv.output_padding[i]
                + (conv.kernel_size[i] - 1) * conv.dilation[i]
                + 1
                - output_shape[i]
            )
            if self.auto_pad == "SAME_UPPER":
                starts.append(total // 2)
                ends.append(total - total // 2)
            else:
                starts.append(total - total // 2)
                ends.append(total // 2)
        return tuple(starts) + tuple(ends)

    def forward(self, input):
        output = self.conv(input)
        pads = self._pads(input.shape[2:])
        ndim = len(pads) // 2
        crop = []
        for i in range(ndim - 1, -1, -1):
            crop.extend([-pads[i], -pads[ndim + i]])
        if any(crop):
            output = torch.nn.functional.pad(output, crop)
        return output
