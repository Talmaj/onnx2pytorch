import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_trilu(x, k=None, upper=None):
    inputs = ["x"]
    initializers = []
    attrs = {}
    if upper is not None:
        attrs["upper"] = upper
    if k is not None:
        inputs.append("k")
        initializers.append(helper.make_tensor("k", TensorProto.INT64, [], [k]))

    node = helper.make_node("Trilu", inputs=inputs, outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "trilu_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    exp_y = ort.InferenceSession(model.SerializeToString()).run(None, {"x": x})[0]

    with torch.no_grad():
        y = ConvertModel(model)(torch.from_numpy(x))

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("upper", [None, 0, 1])
@pytest.mark.parametrize("k", [None, 0, 1, 2, -1, -2])
def test_trilu(upper, k):
    np.random.seed(0)
    x = np.random.randn(4, 5).astype(np.float32)
    check_trilu(x, k, upper)


@pytest.mark.parametrize("upper", [0, 1])
def test_trilu_batched(upper):
    np.random.seed(0)
    x = np.random.randn(2, 3, 4, 4).astype(np.float32)
    check_trilu(x, 1, upper)
