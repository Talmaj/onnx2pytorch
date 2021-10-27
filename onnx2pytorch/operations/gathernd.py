import torch
from torch import nn


class GatherND(nn.Module):
    def __init__(self, batch_dims=0):
        if batch_dims != 0:
            raise NotImplementedError(
                f"GatherND for batch_dims={batch_dims} not implemented."
            )
        self.batch_dims = batch_dims
        super().__init__()

    def forward(self, data: torch.Tensor, indices: torch.Tensor):
        orig_shape = list(indices.shape)
        num_samples = torch.prod(torch.tensor(orig_shape[:-1]))
        m = orig_shape[-1]
        n = len(data.shape)

        if m > n:
            raise ValueError(
                f"The last dimension of indices must be <= the rank of data."
                f"Got indices:{indices.shape}, data:{data.shape}. {m} > {n}"
            )
        out_shape = orig_shape[:-1] + list(data.shape)[m:]

        indices = indices.reshape((num_samples, m)).transpose(0, 1)
        indices = torch.split(indices, 1, 0)
        output = data[indices]  # (num_samples, ...)
        return output.reshape(out_shape).contiguous()
