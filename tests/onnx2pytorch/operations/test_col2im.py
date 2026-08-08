import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_col2im(x, image_shape, block_shape, **attrs):
    node = helper.make_node(
        "Col2Im",
        inputs=["x", "image_shape", "block_shape"],
        outputs=["y"],
        **attrs,
    )
    initializers = [
        helper.make_tensor(
            "image_shape", TensorProto.INT64, [len(image_shape)], image_shape
        ),
        helper.make_tensor(
            "block_shape", TensorProto.INT64, [len(block_shape)], block_shape
        ),
    ]
    graph = helper.make_graph(
        [node],
        "col2im_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_y = ort_session.run(None, {"x": x})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(torch.from_numpy(x))

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-5, atol=1e-5)


def test_col2im():
    np.random.seed(0)
    x = np.random.randn(1, 5 * 1 * 5, 5).astype(np.float32)
    check_col2im(x, [5, 5], [1, 5])


def test_col2im_strides():
    np.random.seed(0)
    x = np.random.randn(1, 1 * 3 * 3, 4).astype(np.float32)
    check_col2im(x, [5, 5], [3, 3], strides=[2, 2])


def test_col2im_pads():
    np.random.seed(0)
    x = np.random.randn(1, 1 * 2 * 2, 24).astype(np.float32)
    check_col2im(x, [5, 5], [2, 2], pads=[0, 1, 0, 1])


def test_col2im_dilations():
    np.random.seed(0)
    x = np.random.randn(1, 1 * 2 * 2, 5).astype(np.float32)
    check_col2im(x, [6, 6], [2, 2], dilations=[1, 5], strides=[1, 1])


def test_col2im_multiple_channels_and_batch():
    np.random.seed(0)
    x = np.random.randn(2, 3 * 2 * 2, 16).astype(np.float32)
    check_col2im(x, [5, 5], [2, 2])
