import torch
from torch import nn
from torchvision.ops import deform_conv2d


class DeformConv(nn.Module):
    def __init__(
        self,
        kernel_size=None,
        dilation=1,
        padding=0,
        stride=1,
        groups=1,
        offset_group=1,
    ):
        super().__init__()
        if isinstance(padding, nn.Module):
            raise NotImplementedError(
                "DeformConv with asymmetric pads not implemented."
            )
        self.kernel_size = kernel_size
        self.dilation = self._pair(dilation)
        self.padding = self._pair(padding)
        self.stride = self._pair(stride)
        self.groups = groups
        self.offset_group = offset_group

    @staticmethod
    def _pair(value):
        if isinstance(value, (tuple, list)):
            return tuple(int(v) for v in value)
        return (int(value), int(value))

    def forward(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        offset: torch.Tensor,
        B: torch.Tensor = None,
        mask: torch.Tensor = None,
    ):
        if X.ndim != 4:
            raise NotImplementedError("DeformConv only implemented for 2D inputs.")
        return deform_conv2d(
            X,
            offset,
            W,
            bias=B,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )

    def extra_repr(self) -> str:
        return "stride={}, padding={}, dilation={}".format(
            self.stride, self.padding, self.dilation
        )
