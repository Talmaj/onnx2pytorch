import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_global_max_pool(x):
    node = helper.make_node("GlobalMaxPool", inputs=["x"], outputs=["y"])
    graph = helper.make_graph(
        [node],
        "globalmaxpool_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_y = ort_session.run(None, {"x": x})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(torch.from_numpy(x))

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "shape", [(1, 3, 5, 5), (2, 4, 3, 7), (2, 3, 6), (1, 2, 3, 4, 5)]
)
def test_global_max_pool(shape):
    np.random.seed(0)
    check_global_max_pool(np.random.randn(*shape).astype(np.float32))
