import numpy as np
import pytest

from tests.onnx2pytorch.differential import (
    assert_matches_oracle,
    make_single_node_model,
)


@pytest.mark.parametrize("transB", [0, 1])
@pytest.mark.parametrize("transA", [0, 1])
@pytest.mark.parametrize("opset_version", [9, 13])
def test_gemm_transpose(opset_version, transA, transB):
    """transA=1 used to raise at every opset."""
    np.random.seed(0)
    a = np.random.randn(*((3, 2) if transA else (2, 3))).astype(np.float32)
    b = np.random.randn(*((4, 3) if transB else (3, 4))).astype(np.float32)
    c = np.random.randn(4).astype(np.float32)
    model = make_single_node_model(
        "Gemm",
        {"a": a},
        opset_version,
        initializers={"b": b, "c": c},
        transA=transA,
        transB=transB,
    )
    assert_matches_oracle(model, {"a": a})


def test_gemm_transpose_with_multipliers():
    np.random.seed(0)
    a = np.random.randn(3, 2).astype(np.float32)
    b = np.random.randn(3, 4).astype(np.float32)
    c = np.random.randn(4).astype(np.float32)
    model = make_single_node_model(
        "Gemm",
        {"a": a},
        13,
        initializers={"b": b, "c": c},
        transA=1,
        alpha=0.5,
        beta=2.0,
    )
    assert_matches_oracle(model, {"a": a})
