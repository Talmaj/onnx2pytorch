import torch
from torch import nn


class SequenceAt(nn.Module):
    def forward(self, input_sequence, position: torch.Tensor):
        return input_sequence[int(position)]
