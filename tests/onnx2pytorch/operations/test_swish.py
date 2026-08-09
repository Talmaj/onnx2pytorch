import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel
from onnx2pytorch.operations.swish import Swish


def build_swish_model(x, **attrs):
    node = helper.make_node("Swish", inputs=["x"], outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "swish_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(x.shape))],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 24)])


@pytest.mark.parametrize("alpha", [None, 1.0, 0.5, 2.0])
def test_swish(alpha):
    np.random.seed(0)
    x = np.random.randn(3, 4, 5).astype(np.float32)

    attrs = {} if alpha is None else {"alpha": alpha}
    model = build_swish_model(x, **attrs)

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_y = ort_session.run(None, {"x": x})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(torch.from_numpy(x))

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-6, atol=1e-6)


def test_swish_matches_silu():
    x = torch.randn(3, 4)
    op = Swish()
    torch.testing.assert_close(op(x), torch.nn.functional.silu(x))
