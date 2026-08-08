import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_compress(x, condition, axis=None):
    attrs = {} if axis is None else {"axis": axis}
    node = helper.make_node(
        "Compress", inputs=["x", "condition"], outputs=["y"], **attrs
    )
    graph = helper.make_graph(
        [node],
        "compress_test",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape)),
            helper.make_tensor_value_info(
                "condition", TensorProto.BOOL, list(condition.shape)
            ),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    exp_y = ort.InferenceSession(model.SerializeToString()).run(
        None, {"x": x, "condition": condition}
    )[0]

    with torch.no_grad():
        y = ConvertModel(model)(torch.from_numpy(x), torch.from_numpy(condition))

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_compress_with_axis(axis):
    np.random.seed(0)
    x = np.random.randn(3, 4).astype(np.float32)
    condition = np.array([True, False, True, False])[: x.shape[axis]]
    check_compress(x, condition, axis)


def test_compress_flattened():
    x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    condition = np.array([False, True, True, False, True, False])
    check_compress(x, condition)


def test_compress_shorter_condition():
    x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    condition = np.array([False, True])
    check_compress(x, condition, 0)


def test_compress_all_false():
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    condition = np.array([False, False])
    check_compress(x, condition, 1)
