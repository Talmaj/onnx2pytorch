import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_lp_pool(x, opset_version=18, **attrs):
    node = helper.make_node("LpPool", inputs=["x"], outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "lppool_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset_version)]
    )

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_y = ort_session.run(None, {"x": x})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(torch.from_numpy(x))

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_lp_pool_2d(p):
    np.random.seed(0)
    x = np.random.randn(2, 3, 6, 6).astype(np.float32)
    check_lp_pool(x, kernel_shape=[2, 2], p=p)


def test_lp_pool_2d_strides():
    np.random.seed(0)
    x = np.random.randn(1, 3, 7, 7).astype(np.float32)
    check_lp_pool(x, kernel_shape=[3, 3], strides=[2, 2])


def test_lp_pool_2d_pads():
    np.random.seed(0)
    x = np.random.randn(1, 3, 5, 5).astype(np.float32)
    check_lp_pool(x, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])


def test_lp_pool_2d_ceil_mode():
    np.random.seed(0)
    x = np.random.randn(1, 1, 5, 5).astype(np.float32)
    check_lp_pool(x, kernel_shape=[2, 2], strides=[2, 2], ceil_mode=1)


@pytest.mark.parametrize("auto_pad", ["SAME_UPPER", "SAME_LOWER", "VALID"])
def test_lp_pool_2d_auto_pad(auto_pad):
    np.random.seed(0)
    x = np.random.randn(1, 2, 5, 5).astype(np.float32)
    check_lp_pool(x, kernel_shape=[3, 3], strides=[2, 2], auto_pad=auto_pad)


def test_lp_pool_1d():
    np.random.seed(0)
    x = np.random.randn(2, 3, 8).astype(np.float32)
    check_lp_pool(x, kernel_shape=[3], strides=[2], p=3)


def test_lp_pool_3d():
    np.random.seed(0)
    x = np.random.randn(1, 2, 4, 4, 4).astype(np.float32)
    check_lp_pool(x, kernel_shape=[2, 2, 2])
