from unittest.mock import MagicMock

import numpy as np
import onnx
import torch
import pytest

from onnx2pytorch.convert.operations import convert_operations
from onnx2pytorch.operations import Hardsigmoid


@pytest.fixture
def x():
    return np.random.randn(3, 4, 5).astype(np.float32)


def test_hardsigmoid(x):
    alpha = 1 / 6
    beta = 1 / 2
    op = Hardsigmoid(alpha=alpha, beta=beta)
    # For pytorch's default values it should use torch's Hardsigmoid
    assert isinstance(op, torch.nn.Hardsigmoid)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.clip(x * alpha + beta, 0, 1)
    out = op(torch.from_numpy(x))
    np.testing.assert_allclose(out, torch.from_numpy(y), rtol=1e-6, atol=1e-6)


def test_hardsigmoid_with_custom_alpha_and_beta(x):
    alpha = 0.2
    beta = 0.5
    op = Hardsigmoid(alpha=alpha, beta=beta)
    assert not isinstance(op, torch.nn.Hardsigmoid)
    y = np.clip(x * alpha + beta, 0, 1)
    out = op(torch.from_numpy(x))
    np.testing.assert_allclose(out, torch.from_numpy(y), rtol=1e-6, atol=1e-6)


def test_hardsigmoid_conversion():
    alpha = np.float32(0.2)
    beta = np.float32(0.5)
    node = onnx.helper.make_node(
        "HardSigmoid",
        inputs=["x"],
        outputs=["y"],
        alpha=alpha,
        beta=beta,
    )

    graph = MagicMock()
    graph.initializers = []
    graph.node = [node]
    converted_ops = list(convert_operations(graph, 10))
    op_id, op_name, op = converted_ops[0]
    assert isinstance(op, Hardsigmoid)
    assert op.alpha == alpha
    assert op.beta == beta
