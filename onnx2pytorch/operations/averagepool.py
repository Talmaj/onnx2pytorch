import torch
from torch import nn

from onnx2pytorch.utils import as_tuple


class AveragePool(nn.Module):
    """AveragePool with count_include_pad=0 over padding that had to be materialised.

    Averaging over the padded input counts the pads, so every output position is
    rescaled by the share of real elements in its window, which is what pooling
    an all-ones tensor padded the same way measures.
    """

    def __init__(self, pool, pad):
        super().__init__()
        self.pool = pool
        self.pad = pad

    def forward(self, input):
        pooled = self.pool(self.pad(input))
        counts = self.pool(self.pad(torch.ones_like(input)))
        return torch.where(counts > 0, pooled / counts, torch.zeros_like(pooled))


class DilatedAvgPool(nn.Module):
    """AveragePool with dilations, which torch's AvgPool does not offer.

    The windows are cut out with unfold at the dilated kernel size and then
    subsampled, as unfold has no dilation of its own.
    """

    def __init__(self, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.kernel_size = tuple(kernel_size)
        ndim = len(self.kernel_size)
        self.stride = as_tuple(stride, ndim)
        self.dilation = as_tuple(dilation, ndim)

    def forward(self, input):
        windows = input
        for i, kernel in enumerate(self.kernel_size):
            size = (kernel - 1) * self.dilation[i] + 1
            windows = windows.unfold(2 + i, size, self.stride[i])
        subsample = tuple(slice(None, None, d) for d in self.dilation)
        windows = windows[(Ellipsis,) + subsample]
        return windows.flatten(-len(self.kernel_size)).mean(-1)
