import torch
from torch import nn


class MaxPool(nn.Module):
    """
    Wrapper that turns torch's pooling indices into ONNX's convention.

    Torch flattens indices per (batch, channel) plane, while ONNX flattens
    them over the whole tensor, channels and batch included.
    """

    def __init__(self, pool, storage_order=0):
        super().__init__()
        self.pool = pool
        self.storage_order = storage_order

    def forward(self, X: torch.Tensor):
        y, indices = self.pool(X)
        spatial = X.shape[2:]

        if self.storage_order:
            coords = []
            remainder = indices
            for size in reversed(spatial):
                coords.append(remainder % size)
                remainder = remainder // size
            column_major = torch.zeros_like(indices)
            stride = 1
            for coord, size in zip(reversed(coords), spatial):
                column_major = column_major + coord * stride
                stride *= size
            indices = column_major

        plane = 1
        for size in spatial:
            plane *= size
        planes = X.shape[0] * X.shape[1]
        offset = torch.arange(planes, device=indices.device) * plane
        shape = list(X.shape[:2]) + [1] * (indices.ndim - 2)
        return y, indices + offset.reshape(shape)
