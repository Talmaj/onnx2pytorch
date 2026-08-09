import torch
from torch import nn


class GatherND(nn.Module):
    def __init__(self, batch_dims=0):
        self.batch_dims = batch_dims
        super().__init__()

    def forward(self, data: torch.Tensor, indices: torch.Tensor):
        b = self.batch_dims
        m = indices.shape[-1]
        if m > data.ndim - b:
            raise ValueError(
                f"The last dimension of indices must be <= the rank of data."
                f"Got indices:{indices.shape}, data:{data.shape}."
            )
        out_shape = list(indices.shape[:-1]) + list(data.shape[b + m :])

        gathered = data.reshape(-1, *data.shape[b:])
        num_batches = gathered.shape[0]
        gathered = gathered.reshape(num_batches, -1, *data.shape[b + m :])
        indices = indices.reshape(num_batches, -1, m)

        flat = torch.zeros(indices.shape[:-1], dtype=torch.long, device=data.device)
        for axis in range(m):
            size = data.shape[b + axis]
            flat = flat * size + indices[..., axis] % size

        batch = torch.arange(num_batches, device=data.device).unsqueeze(1)
        return gathered[batch, flat].reshape(out_shape).contiguous()

    def extra_repr(self) -> str:
        return "batch_dims={}".format(self.batch_dims)
