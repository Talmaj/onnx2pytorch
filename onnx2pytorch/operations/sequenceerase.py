import torch
from torch import nn


class SequenceErase(nn.Module):
    def forward(self, input_sequence, position: torch.Tensor = None):
        output_sequence = list(input_sequence)
        index = -1 if position is None else int(position)
        del output_sequence[index]
        return output_sequence
