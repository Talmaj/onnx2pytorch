import torch
from torch import nn
from torch.nn import functional as F


class MaxUnpool(nn.Module):
    """ONNX MaxUnpool.

    ONNX indices are flattened over the whole tensor, while pytorch expects
    them to be flattened per (batch, channel) plane.
    """

    def __init__(self, kernel_size, padding=0, stride=None):
        super().__init__()
        if isinstance(padding, nn.Module):
            raise NotImplementedError("MaxUnpool with asymmetric pads not implemented.")
        spatial = len(kernel_size)
        self.kernel_size = tuple(kernel_size)
        self.padding = (
            tuple(padding)
            if isinstance(padding, (tuple, list))
            else (padding,) * spatial
        )
        self.stride = tuple(stride) if stride else (1,) * spatial

    def forward(
        self,
        X: torch.Tensor,
        I: torch.Tensor,
        output_shape: torch.Tensor = None,
    ):
        spatial = len(self.kernel_size)
        if output_shape is not None:
            output_size = [int(v) for v in output_shape][-spatial:]
        else:
            output_size = [
                (X.shape[2 + i] - 1) * self.stride[i]
                - 2 * self.padding[i]
                + self.kernel_size[i]
                for i in range(spatial)
            ]

        plane = 1
        for size in output_size:
            plane *= size
        indices = torch.remainder(I.long(), plane)

        unpool = getattr(F, "max_unpool{}d".format(spatial))
        return unpool(
            X,
            indices,
            self.kernel_size,
            self.stride,
            self.padding,
            output_size=output_size,
        )

    def extra_repr(self) -> str:
        return "kernel_size={}, stride={}, padding={}".format(
            self.kernel_size, self.stride, self.padding
        )
