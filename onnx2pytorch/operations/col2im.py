import torch
from torch import nn


class Col2Im(nn.Module):
    """
    ONNX Col2Im, which folds sliding blocks back into an image.

    torch's fold only handles two spatial dimensions and symmetric pads, so the
    blocks are scattered back by hand: every block element is mapped to its flat
    position in the padded image and accumulated there, and the pads are then
    cropped off.
    """

    def __init__(self, dilation=1, padding=0, stride=1):
        super().__init__()
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    @staticmethod
    def _expand(value, spatial):
        if isinstance(value, (tuple, list)):
            return tuple(int(v) for v in value)
        return (int(value),) * spatial

    def _pads(self, spatial):
        """The (before, after) pad per spatial dimension."""
        padding = self.padding
        if isinstance(padding, nn.Module):
            # torch's ConstantPadNd lists the last dimension first
            values = padding.padding
            return [(values[i], values[i + 1]) for i in range(len(values) - 2, -1, -2)]
        return [(p, p) for p in self._expand(padding, spatial)]

    def forward(
        self,
        input: torch.Tensor,
        image_shape: torch.Tensor,
        block_shape: torch.Tensor,
    ):
        image_shape = [int(v) for v in image_shape]
        block_shape = [int(v) for v in block_shape]
        spatial = len(image_shape)
        dilation = self._expand(self.dilation, spatial)
        stride = self._expand(self.stride, spatial)
        pads = self._pads(spatial)

        batch = input.shape[0]
        block_elements = 1
        for size in block_shape:
            block_elements *= size
        channels = input.shape[1] // block_elements

        padded_shape = [
            size + before + after for size, (before, after) in zip(image_shape, pads)
        ]
        blocks_per_dim = [
            (padded - dilation[d] * (block_shape[d] - 1) - 1) // stride[d] + 1
            for d, padded in enumerate(padded_shape)
        ]

        # The flat position of block element o of block p, summed dimension by
        # dimension, gives an index per (block element, block) pair.
        index = torch.zeros(
            block_elements, input.shape[2], dtype=torch.long, device=input.device
        )
        row_stride = 1
        for d in reversed(range(spatial)):
            offsets = torch.arange(block_shape[d], device=input.device) * dilation[d]
            positions = torch.arange(blocks_per_dim[d], device=input.device) * stride[d]
            index += row_stride * (
                self._broadcast(offsets, block_shape, d).reshape(-1, 1)
                + self._broadcast(positions, blocks_per_dim, d).reshape(1, -1)
            )
            row_stride *= padded_shape[d]

        padded = torch.zeros(
            batch,
            channels,
            row_stride,
            dtype=input.dtype,
            device=input.device,
        )
        padded.index_add_(
            2,
            index.reshape(-1),
            input.reshape(batch, channels, -1),
        )
        padded = padded.reshape([batch, channels] + padded_shape)
        for d, (before, _) in enumerate(pads):
            padded = padded.narrow(d + 2, before, image_shape[d])
        return padded

    @staticmethod
    def _broadcast(values, shape, dim):
        """Repeat values so that they vary along dim of a raveled grid."""
        inner = 1
        for size in shape[dim + 1 :]:
            inner *= size
        outer = 1
        for size in shape[:dim]:
            outer *= size
        return values.repeat_interleave(inner).repeat(outer)
