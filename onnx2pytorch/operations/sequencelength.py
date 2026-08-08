import torch
from torch import nn


class SequenceLength(nn.Module):
    def forward(self, input_sequence):
        return torch.tensor(len(input_sequence), dtype=torch.int64)
