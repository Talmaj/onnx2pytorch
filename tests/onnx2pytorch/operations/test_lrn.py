import numpy as np
import pytest

from tests.onnx2pytorch.differential import (
    assert_matches_oracle,
    make_single_node_model,
)


@pytest.fixture
def x():
    np.random.seed(0)
    return np.random.randn(2, 6, 3, 3).astype(np.float32)


@pytest.mark.parametrize("size", [1, 3, 5, 7])
@pytest.mark.parametrize("opset_version", [1, 13])
def test_lrn(opset_version, size, x):
    """Neither size nor bias used to be extracted, so LRN never converted."""
    model = make_single_node_model(
        "LRN", {"x": x}, opset_version, size=size, alpha=0.0002, beta=0.6, bias=2.0
    )
    assert_matches_oracle(model, {"x": x})


def test_lrn_defaults(x):
    model = make_single_node_model("LRN", {"x": x}, 13, size=3)
    assert_matches_oracle(model, {"x": x})


@pytest.mark.parametrize("size", [2, 4])
def test_lrn_even_size(size, x):
    """Onnxruntime rejects even sizes and onnx's reference evaluator is buggy,
    so only check the window placement the spec prescribes."""
    from onnx2pytorch.operations import LRN
    import torch

    y = LRN(alpha=0.0002, beta=0.6, bias=2.0, size=size)(torch.tensor(x))

    channels = x.shape[1]
    square_sum = np.zeros_like(x)
    for c in range(channels):
        begin = max(0, c - (size - 1) // 2)
        end = min(channels, c + size // 2 + 1)
        square_sum[:, c] = np.sum(x[:, begin:end] ** 2, axis=1)
    expected = x / (2.0 + 0.0002 / size * square_sum) ** 0.6

    np.testing.assert_allclose(y.numpy(), expected, rtol=1e-5, atol=1e-6)
