import numpy as np
import torch
import pytest

import onnx

from onnx2pytorch.operations import Cast
from tests.onnx2pytorch.differential import (
    assert_matches_oracle,
    make_single_node_model,
)


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


def test_cast_accepts_saturate():
    """saturate is mapped in extract_attributes but used to reach the constructor."""
    x = np.random.randn(2, 3).astype(np.float32)
    model = make_single_node_model(
        "Cast", {"x": x}, 21, to=onnx.TensorProto.FLOAT16, saturate=1
    )
    assert_matches_oracle(model, {"x": x})


def test_cast_to_float8_without_saturation_is_rejected():
    with pytest.raises(NotImplementedError, match="saturate=0"):
        Cast("float8_e4m3fn", saturate=0)
