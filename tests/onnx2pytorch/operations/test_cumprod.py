import numpy as np
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel
from onnx2pytorch.operations.cumprod import CumProd


def convert_cumprod_model(x, axis, exclusive, reverse):
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
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 26)])
    return ConvertModel(model)


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
    o2p_model = convert_cumprod_model(x, 0, exclusive, reverse)
    with torch.no_grad():
        y = o2p_model(torch.from_numpy(x))
    np.testing.assert_allclose(y.numpy(), np.array(exp_y, dtype=np.float32))


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_cumprod_2d(axis):
    x = np.arange(1, 7).reshape(2, 3).astype(np.float32)
    op = CumProd()
    y = op(torch.from_numpy(x), torch.tensor(axis))
    np.testing.assert_allclose(y.numpy(), np.cumprod(x, axis=axis))


def test_cumprod_exclusive_reverse_2d():
    x = np.arange(1, 7).reshape(2, 3).astype(np.float32)

    op = CumProd(exclusive=1, reverse=1)
    y = op(torch.from_numpy(x), torch.tensor(1))

    exp_y = np.flip(
        np.cumprod(
            np.concatenate(
                [np.ones((2, 1), dtype=np.float32), np.flip(x, 1)[:, :-1]], axis=1
            ),
            axis=1,
        ),
        1,
    )
    np.testing.assert_allclose(y.numpy(), exp_y)
