import torch
from torch import nn

from onnx2pytorch.operations.autopad import AutoPad


def _leading_pads(pad, X):
    """The pad added before each spatial dimension, in onnx dimension order."""
    if pad is None:
        return [0] * (X.dim() - 2)
    if isinstance(pad, AutoPad):
        return [before for before, _ in pad.pads(X)]
    # torch's ConstantPadNd lists the last dimension first, as (before, after)
    padding = pad.padding
    return [padding[i] for i in range(len(padding) - 2, -1, -2)]


class MaxPool(nn.Module):
    """
    Wrapper that turns torch's pooling indices into ONNX's convention.

    Torch flattens indices per (batch, channel) plane, while ONNX flattens
    them over the whole tensor, channels and batch included. Torch also counts
    from the padded plane it pooled over, while ONNX counts from the input, so
    a materialised pad has to be subtracted back out.
    """

    def __init__(self, pool, storage_order=0, pad=None):
        super().__init__()
        self.pool = pool
        self.pad = pad
        self.storage_order = storage_order

    def forward(self, X: torch.Tensor):
        padded = X if self.pad is None else self.pad(X)
        y, indices = self.pool(padded)
        spatial = X.shape[2:]

        # Split the flat index into per dimension coordinates, drop the pad
        # offset, and re-flatten against the unpadded plane.
        coords = []
        remainder = indices
        for size in reversed(padded.shape[2:]):
            coords.append(remainder % size)
            remainder = remainder // size
        coords.reverse()
        coords = [c - before for c, before in zip(coords, _leading_pads(self.pad, X))]

        # onnx flattens row major, or column major when storage_order is set
        order = range(len(spatial))
        if not self.storage_order:
            order = reversed(order)
        indices = torch.zeros_like(indices)
        stride = 1
        for dim in order:
            indices = indices + coords[dim] * stride
            stride *= spatial[dim]

        plane = 1
        for size in spatial:
            plane *= size
        planes = X.shape[0] * X.shape[1]
        offset = torch.arange(planes, device=indices.device) * plane
        shape = list(X.shape[:2]) + [1] * (indices.ndim - 2)
        return y, indices + offset.reshape(shape)
