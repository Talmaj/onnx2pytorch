import numpy as np
import pytest
import torch

from onnx2pytorch.operations.layernorm import LayerNorm
from tests.onnx2pytorch.differential import (
    assert_matches_oracle,
    make_single_node_model,
)


def layernorm_reference(X, scale, bias, axis=-1, epsilon=1e-5):
    """
    Reference implementation of ONNX LayerNormalization.

    Args:
        X: Input tensor
        scale: Scale tensor (gamma)
        bias: Bias tensor (beta), can be None
        axis: The first normalization dimension
        epsilon: Small value to avoid division by zero

    Returns:
        Normalized output
    """
    if axis < 0:
        axis = len(X.shape) + axis

    # Compute mean and variance over dimensions [axis, ..., rank-1]
    axes = tuple(range(axis, len(X.shape)))
    mean = np.mean(X, axis=axes, keepdims=True)
    variance = np.var(X, axis=axes, keepdims=True)

    normalized = (X - mean) / np.sqrt(variance + epsilon)

    Y = normalized * scale
    if bias is not None:
        Y = Y + bias

    return Y


@pytest.mark.parametrize(
    "shape,axis",
    [
        ((2, 3, 4), -1),
        ((2, 3, 4), -2),
        ((2, 3, 4), -3),
        ((2, 3, 4), 0),
        ((2, 3, 4), 1),
        ((2, 3, 4), 2),
        ((5, 10), 1),
        ((2, 3, 8, 8), -2),
    ],
)
def test_layernorm_axis(shape, axis):
    np.random.seed(0)
    X = np.random.randn(*shape).astype(np.float32)
    normalized_shape = shape[axis if axis >= 0 else len(shape) + axis :]
    scale = np.random.randn(*normalized_shape).astype(np.float32)
    bias = np.random.randn(*normalized_shape).astype(np.float32)
    model = make_single_node_model(
        "LayerNormalization", {"X": X, "scale": scale, "B": bias}, 17, axis=axis
    )
    assert_matches_oracle(model, {"X": X, "scale": scale, "B": bias})


def test_layernorm_without_bias():
    np.random.seed(0)
    X = np.random.randn(2, 3, 4).astype(np.float32)
    scale = np.random.randn(4).astype(np.float32)
    model = make_single_node_model("LayerNormalization", {"X": X, "scale": scale}, 17)
    assert_matches_oracle(model, {"X": X, "scale": scale})


def test_layernorm_custom_epsilon():
    np.random.seed(0)
    X = np.random.randn(2, 3, 4).astype(np.float32)
    scale = np.random.randn(4).astype(np.float32)
    model = make_single_node_model(
        "LayerNormalization", {"X": X, "scale": scale}, 17, epsilon=1e-3
    )
    assert_matches_oracle(model, {"X": X, "scale": scale})


def test_layernorm_scale_as_initializer():
    """The normalized shape used to be read off the scale initializer, so a
    scale arriving at runtime raised an IndexError during conversion."""
    np.random.seed(0)
    X = np.random.randn(2, 3, 4).astype(np.float32)
    scale = np.random.randn(4).astype(np.float32)
    bias = np.random.randn(4).astype(np.float32)
    model = make_single_node_model(
        "LayerNormalization", {"X": X}, 17, initializers={"scale": scale, "B": bias}
    )
    assert_matches_oracle(model, {"X": X})


def test_layernorm_optional_outputs():
    """Mean and InvStdDev used to be dropped, only Y was returned."""
    np.random.seed(0)
    X = np.random.randn(2, 3, 4).astype(np.float32)
    scale = np.random.randn(4).astype(np.float32)
    model = make_single_node_model(
        "LayerNormalization",
        {"X": X, "scale": scale},
        17,
        outputs=("Y", "Mean", "InvStdDev"),
    )
    assert_matches_oracle(model, {"X": X, "scale": scale})


@pytest.mark.parametrize("stash_type", [1, 10])
def test_layernorm_stash_type(stash_type):
    """stash_type used to reach the constructor as an unexpected keyword."""
    np.random.seed(0)
    X = np.random.randn(2, 3, 4).astype(np.float16)
    scale = np.random.randn(4).astype(np.float16)
    op = LayerNorm(stash_type=stash_type)
    y, mean, inv_std_dev = op(torch.tensor(X), torch.tensor(scale))
    assert y.dtype == torch.float16
    expected_accumulation = torch.float32 if stash_type == 1 else torch.float16
    assert mean.dtype == expected_accumulation
    assert inv_std_dev.dtype == expected_accumulation
    expected = layernorm_reference(X.astype(np.float32), scale.astype(np.float32), None)
    assert np.allclose(y.numpy().astype(np.float32), expected, rtol=1e-2, atol=1e-2)


def test_layernorm_unsupported_stash_type():
    with pytest.raises(NotImplementedError, match="stash_type"):
        LayerNorm(stash_type=6)


def test_layernorm_axis_parameter_actually_matters():
    """Normalizing over [axis, ..., rank-1] has to follow axis, not the scale."""
    np.random.seed(0)
    X = np.random.randn(2, 3, 4).astype(np.float32)
    scale = np.random.randn(3, 4).astype(np.float32)
    bias = np.random.randn(3, 4).astype(np.float32)

    y_over_two_dims = LayerNorm(axis=-2)(
        torch.tensor(X), torch.tensor(scale), torch.tensor(bias)
    )[0]
    expected = layernorm_reference(X, scale, bias, axis=-2)
    np.testing.assert_allclose(y_over_two_dims.numpy(), expected, rtol=1e-5, atol=1e-6)

    scale_single = np.random.randn(4).astype(np.float32)
    bias_single = np.random.randn(4).astype(np.float32)
    y_over_one_dim = LayerNorm(axis=-1)(
        torch.tensor(X), torch.tensor(scale_single), torch.tensor(bias_single)
    )[0]
    expected = layernorm_reference(X, scale_single, bias_single, axis=-1)
    np.testing.assert_allclose(y_over_one_dim.numpy(), expected, rtol=1e-5, atol=1e-6)

    assert not np.allclose(y_over_two_dims.numpy(), y_over_one_dim.numpy())


def test_layernorm_negative_axis_equivalence():
    np.random.seed(0)
    X = np.random.randn(2, 3, 4, 5).astype(np.float32)
    scale = np.random.randn(4, 5).astype(np.float32)
    negative = LayerNorm(axis=-2)(torch.tensor(X), torch.tensor(scale))[0]
    positive = LayerNorm(axis=2)(torch.tensor(X), torch.tensor(scale))[0]
    assert torch.allclose(negative, positive)
