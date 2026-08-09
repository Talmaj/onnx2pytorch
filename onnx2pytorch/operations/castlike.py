import torch
from torch import nn

from onnx2pytorch.operations.cast import (
    cast_from_string,
    cast_to_string,
    is_string_array,
)


class CastLike(nn.Module):
    """ONNX CastLike: cast the input to the data type of the target tensor."""

    def forward(self, input: torch.Tensor, target_type: torch.Tensor):
        if is_string_array(target_type):
            return cast_to_string(input)
        elif is_string_array(input):
            return cast_from_string(input, target_type.dtype)
        return input.to(target_type.dtype)
