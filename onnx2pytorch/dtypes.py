"""ONNX to PyTorch data type mappings."""

import torch


# ONNX TensorProto DataType to PyTorch dtype mapping
# Reference: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
ONNX_DTYPE_TO_TORCH = {
    0: None,  # UNDEFINED
    1: torch.float32,  # FLOAT
    2: torch.uint8,  # UINT8
    3: torch.int8,  # INT8
    4: None,  # UINT16 - not supported in PyTorch
    5: torch.int16,  # INT16
    6: torch.int32,  # INT32
    7: torch.int64,  # INT64
    8: None,  # STRING - not supported as tensor dtype
    9: torch.bool,  # BOOL
    10: torch.float16,  # FLOAT16
    11: torch.float64,  # DOUBLE
    12: None,  # UINT32 - not supported in PyTorch
    13: None,  # UINT64 - not supported in PyTorch
    14: torch.complex64,  # COMPLEX64
    15: torch.complex128,  # COMPLEX128
    16: torch.bfloat16,  # BFLOAT16
    17: (
        torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else None
    ),  # FLOAT8E4M3FN
    18: (
        torch.float8_e4m3fnuz if hasattr(torch, "float8_e4m3fnuz") else None
    ),  # FLOAT8E4M3FNUZ
    19: torch.float8_e5m2 if hasattr(torch, "float8_e5m2") else None,  # FLOAT8E5M2
    20: (
        torch.float8_e5m2fnuz if hasattr(torch, "float8_e5m2fnuz") else None
    ),  # FLOAT8E5M2FNUZ
    21: None,  # UINT4 - not supported in PyTorch
    22: None,  # INT4 - not supported in PyTorch
    23: None,  # FLOAT4E2M1 - not supported in PyTorch
    24: None,  # FLOAT8E8M0 - not supported in PyTorch
}
