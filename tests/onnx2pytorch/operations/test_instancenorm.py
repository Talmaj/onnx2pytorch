import numpy as np
import torch
import pytest

from onnx2pytorch.operations import InstanceNormWrapper


def instancenorm_reference(x, s, bias, eps):
    dims_x = len(x.shape)
    axis = tuple(range(2, dims_x))
    mean = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)
    dim_ones = (1,) * (dims_x - 2)
    s = s.reshape(-1, *dim_ones)
    bias = bias.reshape(-1, *dim_ones)
    return s * (x - mean) / np.sqrt(var + eps) + bias


@pytest.fixture
def x_np():
    # input size: (1, 2, 1, 3)
    return np.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(np.float32)


@pytest.fixture
def s_np():
    return np.array([1.0, 1.5]).astype(np.float32)


@pytest.fixture
def b_np():
    return np.array([0, 1]).astype(np.float32)


def test_instancenorm(x_np, s_np, b_np):
    eps = 1e-5
    x = torch.from_numpy(x_np)
    s = torch.from_numpy(s_np)
    b = torch.from_numpy(b_np)

    exp_y = instancenorm_reference(x_np, s_np, b_np, eps).astype(np.float32)
    exp_y_shape = (1, 2, 1, 3)
    op = InstanceNormWrapper([s, b], eps=eps)
    y = op(x)

    assert y.shape == exp_y_shape
    assert np.allclose(y.detach().numpy(), exp_y, rtol=1e-5, atol=1e-5)


def test_instancenorm_lazy(x_np, s_np, b_np):
    eps = 1e-5
    x = torch.from_numpy(x_np)
    s = torch.from_numpy(s_np)
    b = torch.from_numpy(b_np)

    exp_y = instancenorm_reference(x_np, s_np, b_np, eps).astype(np.float32)
    exp_y_shape = (1, 2, 1, 3)
    op = InstanceNormWrapper([], eps=eps)
    y = op(x, s, b)

    assert y.shape == exp_y_shape
    assert np.allclose(y.detach().numpy(), exp_y, rtol=1e-5, atol=1e-5)
