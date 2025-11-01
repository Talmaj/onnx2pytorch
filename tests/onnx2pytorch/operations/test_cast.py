import numpy as np
import torch
import pytest
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

from onnx2pytorch.operations import Cast


@pytest.mark.parametrize("dtype", ["double", "float32", "float16"])
def test_cast(dtype):
    shape = (3, 4)
    x_np = np.array(
        [
            "0.47892547",
            "0.48033667",
            "0.49968487",
            "0.81910545",
            "0.47031248",
            "0.816468",
            "0.21087195",
            "0.7229038",
            "NaN",
            "INF",
            "+INF",
            "-INF",
        ],
        dtype=np.dtype(object),
    ).reshape(shape)
    x = torch.tensor(x_np.astype(dtype))
    op = Cast(dtype)
    y = x_np.astype(getattr(np, dtype.lower()))
    assert np.allclose(op(x).numpy(), y, rtol=0, atol=0, equal_nan=True)
