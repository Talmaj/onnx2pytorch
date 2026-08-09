import numpy as np
import torch
from torch import nn

STRING_KINDS = ("O", "S", "U")


def is_string_array(value):
    return isinstance(value, np.ndarray) and value.dtype.kind in STRING_KINDS


def cast_to_string(input):
    x = input.numpy() if torch.is_tensor(input) else np.asarray(input)
    if x.dtype.kind not in STRING_KINDS + ("i", "u"):
        raise NotImplementedError(
            "Cast from {} to string not implemented, numpy does not format the "
            "values the way onnx specifies.".format(x.dtype)
        )
    return x.astype(np.str_).astype(object)


def cast_from_string(input, dtype):
    if isinstance(dtype, torch.dtype):
        dtype = torch.empty(0, dtype=dtype).numpy().dtype
    return torch.from_numpy(np.asarray(input).astype(np.str_).astype(dtype))


class Cast(nn.Module):
    def __init__(self, dtype, saturate=1):
        super().__init__()
        if isinstance(dtype, str) and dtype.lower() != "string":
            dtype = getattr(torch, dtype.lower())
        if not saturate and "float8" in str(dtype):
            raise NotImplementedError(
                "Cast to {} with saturate=0 not implemented.".format(dtype)
            )
        self.dtype = dtype

    def forward(self, input):
        if self.dtype == "string":
            return cast_to_string(input)
        elif is_string_array(input):
            return cast_from_string(input, self.dtype)
        return input.to(self.dtype)

    def extra_repr(self) -> str:
        return "dtype={}".format(self.dtype)
