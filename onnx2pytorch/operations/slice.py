import torch
from torch import nn


def _to_positive_step(orig_slice, N):
    """
    Convert a slice object with a negative step to one with a positive step.
    Accessing an iterable with the positive-stepped slice, followed by flipping
    the result, should be equivalent to accessing the tensor with the original
    slice. Computing positive-step slice requires using N, the length of the
    iterable being sliced. This is because PyTorch currently does not support
    slicing a tensor with a negative step.
    """
    # Get rid of backward slices
    start, stop, step = orig_slice.indices(N)

    # Get number of steps and remainder
    n, r = divmod(stop - start, step)
    if n < 0 or (n == 0 and r == 0):
        return slice(0, 0, 1)
    if r != 0:  # a "stop" index, not a last index
        n += 1

    if step < 0:
        start, stop, step = start + (n - 1) * step, start - step, -step
    else:  # step > 0, step == 0 is not allowed
        stop = start + n * step
    stop = min(stop, N)

    return slice(start, stop, step)


class Slice(nn.Module):
    def __init__(self, dim=None, starts=None, ends=None, steps=None):
        self.dim = [dim] if isinstance(dim, int) else dim
        self.starts = starts
        self.ends = ends
        self.steps = steps
        super().__init__()

    def forward(
        self, data: torch.Tensor, starts=None, ends=None, axes=None, steps=None
    ):
        if axes is None:
            axes = self.dim
        if starts is None:
            starts = self.starts
        if ends is None:
            ends = self.ends
        if steps is None:
            steps = self.steps

        if isinstance(starts, (tuple, list)):
            starts = torch.tensor(starts, device=data.device)
        if isinstance(ends, (tuple, list)):
            ends = torch.tensor(ends, device=data.device)
        if isinstance(steps, (tuple, list)):
            steps = torch.tensor(steps, device=data.device)

        # If axes=None set them to (0, 1, 2, ...)
        if axes is None:
            axes = tuple(torch.arange(len(starts)))
        if steps is None:
            steps = tuple(torch.tensor(1) for _ in axes)

        axes = [data.ndim + x if x < 0 else x for x in axes]

        selection = [slice(None) for _ in range(max(axes) + 1)]

        flip_dims = []
        for i, axis in enumerate(axes):
            raw_slice = slice(
                starts[i].to(dtype=torch.long, device=data.device),
                ends[i].to(dtype=torch.long, device=data.device),
                steps[i].to(dtype=torch.long, device=data.device),
            )
            if steps[i] < 0:
                selection[axis] = _to_positive_step(raw_slice, data.shape[axis])
                flip_dims.append(axis)
            else:
                selection[axis] = raw_slice
        if len(flip_dims) > 0:
            return torch.flip(data.__getitem__(selection), flip_dims)
        else:
            # For torch < 1.8.1, torch.flip cannot handle empty dims
            return data.__getitem__(selection)
