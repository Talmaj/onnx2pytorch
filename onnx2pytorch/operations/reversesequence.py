import torch
from torch import nn


class ReverseSequence(nn.Module):
    def __init__(self, batch_axis=1, time_axis=0):
        super().__init__()
        self.batch_axis = batch_axis
        self.time_axis = time_axis

    def forward(self, input: torch.Tensor, sequence_lens: torch.Tensor):
        out = input.clone()
        for batch, length in enumerate(sequence_lens.tolist()):
            source = [slice(None)] * input.ndim
            target = [slice(None)] * input.ndim
            source[self.batch_axis] = batch
            target[self.batch_axis] = batch
            source[self.time_axis] = torch.arange(
                length - 1, -1, -1, device=input.device
            )
            target[self.time_axis] = torch.arange(length, device=input.device)
            out[tuple(target)] = input[tuple(source)]
        return out

    def extra_repr(self) -> str:
        return "batch_axis={}, time_axis={}".format(self.batch_axis, self.time_axis)
