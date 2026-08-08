import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_cumsum(x, axis, exclusive, reverse):
    node = helper.make_node(
        "CumSum",
        inputs=["x", "axis"],
        outputs=["y"],
        exclusive=exclusive,
        reverse=reverse,
    )
    graph = helper.make_graph(
        [node],
        "cumsum_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor("axis", TensorProto.INT64, [], [axis])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_y = ort_session.run(None, {"x": x})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(torch.from_numpy(x))

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "exclusive, reverse",
    [(0, 0), (1, 0), (0, 1), (1, 1)],
)
def test_cumsum_1d(exclusive, reverse):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    check_cumsum(x, 0, exclusive, reverse)


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("exclusive, reverse", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_cumsum_2d(axis, exclusive, reverse):
    x = np.arange(1, 7).reshape(2, 3).astype(np.float32)
    check_cumsum(x, axis, exclusive, reverse)


def test_cumsum_3d():
    np.random.seed(0)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    check_cumsum(x, 2, 0, 0)
