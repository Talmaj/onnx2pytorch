import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_blackman_window(size, out_type=TensorProto.FLOAT, **attrs):
    node = helper.make_node("BlackmanWindow", inputs=["size"], outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "blackmanwindow_test",
        [helper.make_tensor_value_info("size", TensorProto.INT64, [])],
        [helper.make_tensor_value_info("y", out_type, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    feed = {"size": np.array(size, dtype=np.int64)}
    exp_y = ort.InferenceSession(model.SerializeToString()).run(None, feed)[0]
    with torch.no_grad():
        y = ConvertModel(model)(torch.from_numpy(feed["size"]))
    assert y.numpy().dtype == exp_y.dtype
    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("size", [1, 2, 10, 33])
def test_blackman_window(size):
    check_blackman_window(size)


@pytest.mark.parametrize("periodic", [0, 1])
def test_blackman_window_periodic(periodic):
    check_blackman_window(16, periodic=periodic)


def test_blackman_window_double():
    check_blackman_window(
        12, out_type=TensorProto.DOUBLE, output_datatype=TensorProto.DOUBLE
    )
