import numpy as np
import pytest
import torch
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

from onnx2pytorch.convert import ConvertModel


def build_model(x, axis, exclusive=0, reverse=0):
    node = helper.make_node(
        "CumProd",
        inputs=["x", "axis"],
        outputs=["y"],
        exclusive=exclusive,
        reverse=reverse,
    )
    graph = helper.make_graph(
        [node],
        "cumprod_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor("axis", TensorProto.INT64, [], [axis])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 26)])


def check_cumprod(x, axis, exclusive=0, reverse=0):
    model = build_model(x, axis, exclusive, reverse)

    # onnxruntime 1.28 has no CumProd kernel
    exp_y = ReferenceEvaluator(model).run(None, {"x": x})[0]

    with torch.no_grad():
        y = ConvertModel(model)(torch.from_numpy(x))
    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "exclusive, reverse, exp_y",
    [
        (0, 0, [1.0, 2.0, 6.0]),
        (1, 0, [1.0, 1.0, 2.0]),
        (0, 1, [6.0, 6.0, 3.0]),
        (1, 1, [6.0, 3.0, 1.0]),
    ],
)
def test_cumprod_1d(exclusive, reverse, exp_y):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    model = build_model(x, 0, exclusive, reverse)
    with torch.no_grad():
        y = ConvertModel(model)(torch.from_numpy(x))
    np.testing.assert_allclose(y.numpy(), np.array(exp_y, dtype=np.float32))
    check_cumprod(x, 0, exclusive, reverse)


@pytest.mark.parametrize("reverse", [0, 1])
@pytest.mark.parametrize("exclusive", [0, 1])
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_cumprod_2d(axis, exclusive, reverse):
    x = np.arange(1, 7).reshape(2, 3).astype(np.float32)
    check_cumprod(x, axis, exclusive, reverse)


def test_cumprod_3d():
    np.random.seed(0)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    check_cumprod(x, -2, exclusive=1, reverse=1)
