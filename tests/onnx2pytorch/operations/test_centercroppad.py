import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_center_crop_pad(x, shape, axes=None):
    attrs = {} if axes is None else {"axes": axes}
    node = helper.make_node(
        "CenterCropPad", inputs=["x", "shape"], outputs=["y"], **attrs
    )
    initializers = [helper.make_tensor("shape", TensorProto.INT64, [len(shape)], shape)]
    graph = helper.make_graph(
        [node],
        "centercroppad_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    exp_y = ort.InferenceSession(model.SerializeToString()).run(None, {"x": x})[0]

    with torch.no_grad():
        y = ConvertModel(model)(torch.from_numpy(x))

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-5, atol=1e-5)


def test_center_crop_pad_crop():
    np.random.seed(0)
    x = np.random.randn(20, 10, 3).astype(np.float32)
    check_center_crop_pad(x, [10, 7, 3])


def test_center_crop_pad_pad():
    np.random.seed(0)
    x = np.random.randn(10, 7, 3).astype(np.float32)
    check_center_crop_pad(x, [20, 10, 3])


def test_center_crop_pad_crop_and_pad():
    np.random.seed(0)
    x = np.random.randn(20, 8, 3).astype(np.float32)
    check_center_crop_pad(x, [10, 10, 3])


@pytest.mark.parametrize("axes", [[0], [1], [0, 1], [-2, -1]])
def test_center_crop_pad_axes(axes):
    np.random.seed(0)
    x = np.random.randn(9, 11, 5).astype(np.float32)
    shape = [6] * len(axes)
    check_center_crop_pad(x, shape, axes)
