import torch
from torch import nn


class TensorScatter(nn.Module):
    """ONNX TensorScatter: write an update into a cache along the sequence dimension."""

    def __init__(self, dim=-2, mode="linear"):
        super().__init__()
        if mode not in ("linear", "circular"):
            raise NotImplementedError(
                "TensorScatter with mode={} not implemented.".format(mode)
            )
        self.dim = dim
        self.mode = mode

    def forward(
        self,
        past_cache: torch.Tensor,
        update: torch.Tensor,
        write_indices: torch.Tensor = None,
    ):
        dim = self.dim % past_cache.ndim
        sequence_length = update.shape[dim]
        batch_size = past_cache.shape[0]

        if write_indices is None:
            write_indices = torch.zeros(
                batch_size, dtype=torch.long, device=past_cache.device
            )
        positions = write_indices.to(torch.long).reshape(-1, 1) + torch.arange(
            sequence_length, device=past_cache.device
        )
        if self.mode == "circular":
            positions = positions % past_cache.shape[dim]

        shape = [batch_size] + [1] * (past_cache.ndim - 1)
        shape[dim] = sequence_length
        index = positions.reshape(shape).expand(update.shape)
        return past_cache.scatter(dim, index, update)
