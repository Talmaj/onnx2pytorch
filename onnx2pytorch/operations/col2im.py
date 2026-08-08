import torch
from torch import nn
from torch.nn import functional as F


class Col2Im(nn.Module):
    def __init__(self, dilation=1, padding=0, stride=1):
        super().__init__()
        if isinstance(padding, nn.Module):
            raise NotImplementedError("Col2Im with asymmetric pads not implemented.")
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    @staticmethod
    def _expand(value, spatial):
        if isinstance(value, (tuple, list)):
            return tuple(int(v) for v in value)
        return (int(value),) * spatial

    def forward(
        self,
        input: torch.Tensor,
        image_shape: torch.Tensor,
        block_shape: torch.Tensor,
    ):
        image_shape = [int(v) for v in image_shape]
        block_shape = [int(v) for v in block_shape]
        spatial = len(image_shape)
        if spatial != 2:
            raise NotImplementedError(
                "Col2Im not implemented for {} spatial dimensions.".format(spatial)
            )
        return F.fold(
            input,
            output_size=image_shape,
            kernel_size=block_shape,
            dilation=self._expand(self.dilation, spatial),
            padding=self._expand(self.padding, spatial),
            stride=self._expand(self.stride, spatial),
        )
