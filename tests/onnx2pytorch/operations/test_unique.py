import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_unique(x, num_outputs=4, axis=None, sorted=None):
    attrs = {}
    if axis is not None:
        attrs["axis"] = axis
    if sorted is not None:
        attrs["sorted"] = sorted

    names = ["y", "indices", "inverse_indices", "counts"][:num_outputs]
    node = helper.make_node("Unique", inputs=["x"], outputs=names, **attrs)
    graph_outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)]
    for name in names[1:]:
        graph_outputs.append(
            helper.make_tensor_value_info(name, TensorProto.INT64, None)
        )
    graph = helper.make_graph(
        [node],
        "unique_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        graph_outputs,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    exp = ort.InferenceSession(model.SerializeToString()).run(None, {"x": x})

    with torch.no_grad():
        res = ConvertModel(model)(torch.from_numpy(x))
    if num_outputs == 1:
        res = [res]

    for actual, expected in zip(res, exp):
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("sorted", [None, 0, 1])
def test_unique_flattened(sorted):
    x = np.array([2.0, 1.0, 1.0, 3.0, 4.0, 3.0], dtype=np.float32)
    check_unique(x, sorted=sorted)


@pytest.mark.parametrize("sorted", [0, 1])
def test_unique_flattened_2d_input(sorted):
    x = np.array([[3.0, 1.0], [1.0, 2.0], [3.0, 1.0]], dtype=np.float32)
    check_unique(x, sorted=sorted)


@pytest.mark.parametrize("sorted", [0, 1])
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_unique_with_axis(axis, sorted):
    x = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 3.0, 4.0]], dtype=np.float32)
    check_unique(x, axis=axis, sorted=sorted)


@pytest.mark.parametrize("num_outputs", [1, 2, 3, 4])
def test_unique_partial_outputs(num_outputs):
    x = np.array([5.0, 4.0, 4.0, 5.0, 1.0], dtype=np.float32)
    check_unique(x, num_outputs=num_outputs)


def test_unique_all_distinct():
    np.random.seed(0)
    x = np.random.permutation(10).astype(np.float32)
    check_unique(x, sorted=0)
