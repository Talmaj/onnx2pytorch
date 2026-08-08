import torch
from torch import nn
from torch.nn import functional as F

from onnx2pytorch.operations.autopad import AutoPad


class LpPool(nn.Module):
    def __init__(
        self,
        kernel_size,
        p=2,
        stride=None,
        padding=0,
        ceil_mode=False,
        dilation=1,
        auto_pad=None,
    ):
        super().__init__()
        self.kernel_size = tuple(kernel_size)
        self.p = p
        self.stride = tuple(stride) if stride else (1,) * len(self.kernel_size)
        self.ceil_mode = bool(ceil_mode)

        dilations = dilation if isinstance(dilation, (tuple, list)) else (dilation,)
        if any(d != 1 for d in dilations):
            raise NotImplementedError("LpPool with dilations != 1 not implemented.")

        self.padding = (0,) * len(self.kernel_size)
        self.pad_layer = None
        if auto_pad is not None:
            self.pad_layer = AutoPad(
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=1,
                mode=auto_pad,
            )
        elif isinstance(padding, nn.Module):
            self.pad_layer = padding
        elif isinstance(padding, (tuple, list)):
            self.padding = tuple(padding)
        elif padding:
            self.padding = (padding,) * len(self.kernel_size)

    def forward(self, input: torch.Tensor):
        x = torch.abs(input) ** self.p
        if self.pad_layer is not None:
            x = self.pad_layer(x)

        spatial = len(self.kernel_size)
        if spatial == 1:
            pooled = F.avg_pool2d(
                x.unsqueeze(-2),
                (1,) + self.kernel_size,
                (1,) + self.stride,
                (0,) + self.padding,
                ceil_mode=self.ceil_mode,
                divisor_override=1,
            ).squeeze(-2)
        elif spatial == 2:
            pooled = F.avg_pool2d(
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                ceil_mode=self.ceil_mode,
                divisor_override=1,
            )
        elif spatial == 3:
            pooled = F.avg_pool3d(
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                ceil_mode=self.ceil_mode,
                divisor_override=1,
            )
        else:
            raise NotImplementedError(
                "LpPool not implemented for {} spatial dimensions.".format(spatial)
            )
        return pooled ** (1.0 / self.p)

    def extra_repr(self) -> str:
        return "kernel_size={}, p={}, stride={}".format(
            self.kernel_size, self.p, self.stride
        )
