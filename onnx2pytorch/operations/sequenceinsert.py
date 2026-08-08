import torch
from torch import nn


class SequenceInsert(nn.Module):
    def forward(
        self,
        input_sequence,
        tensor: torch.Tensor,
        position: torch.Tensor = None,
    ):
        output_sequence = list(input_sequence)
        if position is None:
            output_sequence.append(tensor)
            return output_sequence
        index = int(position)
        if index < 0:
            index += len(output_sequence)
        output_sequence.insert(index, tensor)
        return output_sequence
