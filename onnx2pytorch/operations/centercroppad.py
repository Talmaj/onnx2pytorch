import torch
from torch import nn


class CenterCropPad(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        if isinstance(dim, int):
            dim = (dim,)
        self.dim = dim

    def forward(self, input_data: torch.Tensor, shape: torch.Tensor):
        axes = self.dim
        if axes is None:
            axes = range(input_data.ndim)
        axes = [a if a >= 0 else a + input_data.ndim for a in axes]

        out_shape = list(input_data.shape)
        for axis, size in zip(axes, [int(v) for v in shape]):
            out_shape[axis] = size

        source = []
        target = []
        for axis, size in enumerate(out_shape):
            dim_size = input_data.shape[axis]
            if dim_size > size:
                start = (dim_size - size) // 2
                source.append(slice(start, start + size))
                target.append(slice(0, size))
            elif dim_size < size:
                start = (size - dim_size) // 2
                source.append(slice(0, dim_size))
                target.append(slice(start, start + dim_size))
            else:
                source.append(slice(None))
                target.append(slice(None))

        out = torch.zeros(out_shape, dtype=input_data.dtype, device=input_data.device)
        out[tuple(target)] = input_data[tuple(source)]
        return out

    def extra_repr(self) -> str:
        return "dim={}".format(self.dim)
