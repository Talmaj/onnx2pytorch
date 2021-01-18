import torch
from torch import nn


class Slice(nn.Module):
    def forward(self, input: torch.Tensor, starts, ends, axes=None, steps=None):
        # If axes=None set them to (0, 1, 2, ...)
        if axes is None:
            axes = tuple(range(len(starts)))
        if steps is None:
            steps = tuple(1 for _ in axes)

        selection = [slice(None) for _ in range(max(axes) + 1)]
        for i, axis in enumerate(axes):
            selection[axis] = slice(starts[i], ends[i], steps[i])
        return input.__getitem__(selection)
