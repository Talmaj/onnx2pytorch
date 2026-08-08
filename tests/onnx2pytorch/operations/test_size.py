import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 3, 4), (1, 1, 1, 1)])
def test_size(shape):
    np.random.seed(0)
    x = np.random.randn(*shape).astype(np.float32)

    node = helper.make_node("Size", inputs=["x"], outputs=["y"])
    graph = helper.make_graph(
        [node],
        "size_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.INT64, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    exp_y = ort.InferenceSession(model.SerializeToString()).run(None, {"x": x})[0]

    with torch.no_grad():
        y = ConvertModel(model)(torch.from_numpy(x))

    assert y.dtype == torch.int64
    np.testing.assert_array_equal(y.numpy(), exp_y)
