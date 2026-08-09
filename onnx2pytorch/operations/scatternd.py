import torch
from torch import nn


class ScatterND(nn.Module):
    def __init__(self, reduction="none"):
        super().__init__()
        self.reduction = reduction

    def forward(self, data: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor):
        output = data.clone()
        k = indices.shape[-1]
        flat_indices = indices.reshape(-1, k).long()
        flat_updates = updates.reshape((-1,) + tuple(data.shape[k:]))

        index = []
        for i in range(k):
            column = flat_indices[:, i]
            index.append(torch.where(column < 0, column + data.shape[i], column))
        index = tuple(index)

        if self.reduction == "none":
            output[index] = flat_updates
        elif self.reduction == "add":
            output.index_put_(index, flat_updates, accumulate=True)
        else:
            # Duplicate indices have to be folded in one at a time
            combine = {
                "mul": torch.mul,
                "max": torch.maximum,
                "min": torch.minimum,
            }[self.reduction]
            for position, update in zip(zip(*index), flat_updates):
                output[position] = combine(output[position], update)
        return output

    def extra_repr(self) -> str:
        return "reduction={}".format(self.reduction)
