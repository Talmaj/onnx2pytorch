import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_gather_elements(data, indices, axis=None):
    attrs = {} if axis is None else {"axis": axis}
    node = helper.make_node(
        "GatherElements", inputs=["data", "indices"], outputs=["y"], **attrs
    )
    graph = helper.make_graph(
        [node],
        "gatherelements_test",
        [
            helper.make_tensor_value_info("data", TensorProto.FLOAT, list(data.shape)),
            helper.make_tensor_value_info(
                "indices", TensorProto.INT64, list(indices.shape)
            ),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    exp_y = ort.InferenceSession(model.SerializeToString()).run(
        None, {"data": data, "indices": indices}
    )[0]

    with torch.no_grad():
        y = ConvertModel(model)(torch.from_numpy(data), torch.from_numpy(indices))

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-5, atol=1e-5)


def test_gather_elements_axis0():
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    indices = np.array([[0, 0], [1, 0]], dtype=np.int64)
    check_gather_elements(data, indices, 0)


def test_gather_elements_axis1():
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    indices = np.array([[0, 0], [1, 0]], dtype=np.int64)
    check_gather_elements(data, indices, 1)


def test_gather_elements_default_axis():
    np.random.seed(0)
    data = np.random.randn(3, 4).astype(np.float32)
    indices = np.random.randint(0, 3, size=(2, 4)).astype(np.int64)
    check_gather_elements(data, indices)


@pytest.mark.parametrize("axis", [0, 1, 2, -1, -3])
def test_gather_elements_3d(axis):
    np.random.seed(0)
    data = np.random.randn(3, 4, 5).astype(np.float32)
    indices = np.random.randint(0, 3, size=(3, 4, 5)).astype(np.int64)
    check_gather_elements(data, indices, axis)


def test_gather_elements_negative_indices():
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    indices = np.array([[-1, -2], [-2, -1]], dtype=np.int64)
    check_gather_elements(data, indices, 0)
