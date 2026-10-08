"""Conv and ConvTranspose whose weight is computed at runtime."""

from torch import nn
from torch.nn import functional as F

from onnx2pytorch.operations.convtranspose import crop_conv_transpose


class DynamicConv(nn.Module):
    """Functional convolution. Weight (and bias, if present) are inputs."""

    def __init__(
        self,
        ndim,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        transposed=False,
        output_padding=0,
        pad_layer=None,
        transpose_pads=None,
        output_shape=None,
        auto_pad="NOTSET",
        kernel_size=None,
    ):
        super().__init__()
        self.ndim = ndim
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.transposed = transposed
        self.output_padding = output_padding
        self.kernel_size = None if kernel_size is None else tuple(kernel_size)
        self.pad_layer = pad_layer
        self.transpose_pads = None if transpose_pads is None else tuple(transpose_pads)
        self.output_shape = None if output_shape is None else tuple(output_shape)
        self.auto_pad = auto_pad

    def _conv(self, ndim):
        name = "conv_transpose{}d" if self.transposed else "conv{}d"
        try:
            return getattr(F, name.format(ndim))
        except AttributeError:
            raise NotImplementedError("Conv for {} spatial dimensions.".format(ndim))

    def forward(self, input, weight, bias=None):
        if self.pad_layer is not None:
            input = self.pad_layer(input)
        ndim = self.ndim
        if ndim is None:
            ndim = weight.dim() - 2
        conv = self._conv(ndim)
        kwargs = {
            "stride": self.stride,
            "padding": self.padding,
            "dilation": self.dilation,
            "groups": self.groups,
        }
        if self.transposed:
            kwargs["output_padding"] = self.output_padding
        output = conv(input, weight, bias, **kwargs)
        if not self.transposed:
            return output
        kernel = self.kernel_size or tuple(weight.shape[2:])
        return crop_conv_transpose(
            output,
            input.shape[2:],
            self.stride,
            kernel,
            self.dilation,
            self.output_padding,
            self.transpose_pads,
            self.output_shape,
            self.auto_pad,
        )
