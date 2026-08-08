import numpy as np
import onnxruntime as ort
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_max_unpool(x, indices, output_shape=None, **attrs):
    inputs = ["x", "i"]
    initializers = []
    if output_shape is not None:
        inputs.append("output_shape")
        initializers.append(
            helper.make_tensor(
                "output_shape", TensorProto.INT64, [len(output_shape)], output_shape
            )
        )

    node = helper.make_node("MaxUnpool", inputs=inputs, outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "maxunpool_test",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape)),
            helper.make_tensor_value_info("i", TensorProto.INT64, list(indices.shape)),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_y = ort_session.run(None, {"x": x, "i": indices})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(torch.from_numpy(x), torch.from_numpy(indices))

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-5, atol=1e-5)


def test_max_unpool_without_output_shape():
    x = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
    indices = np.array([[[[5, 7], [13, 15]]]], dtype=np.int64)
    check_max_unpool(x, indices, kernel_shape=[2, 2], strides=[2, 2])


def test_max_unpool_with_output_shape():
    x = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
    indices = np.array([[[[5, 7], [13, 15]]]], dtype=np.int64)
    check_max_unpool(
        x, indices, output_shape=[1, 1, 5, 5], kernel_shape=[2, 2], strides=[2, 2]
    )


def test_max_unpool_with_pads():
    x = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=np.float32)
    indices = np.array([[[[1, 3, 4], [6, 8, 9], [11, 13, 14]]]], dtype=np.int64)
    check_max_unpool(x, indices, kernel_shape=[2, 2], strides=[2, 2], pads=[1, 1, 1, 1])


def test_max_unpool_1d():
    x = np.array([[[1, 2, 3]]], dtype=np.float32)
    indices = np.array([[[0, 2, 4]]], dtype=np.int64)
    check_max_unpool(x, indices, kernel_shape=[2], strides=[2])


def test_max_unpool_roundtrip_with_maxpool():
    np.random.seed(0)
    x = np.random.randn(2, 3, 4, 4).astype(np.float32)

    pool = helper.make_node(
        "MaxPool",
        inputs=["x"],
        outputs=["pooled", "indices"],
        kernel_shape=[2, 2],
        strides=[2, 2],
    )
    unpool = helper.make_node(
        "MaxUnpool",
        inputs=["pooled", "indices"],
        outputs=["y"],
        kernel_shape=[2, 2],
        strides=[2, 2],
    )
    graph = helper.make_graph(
        [pool, unpool],
        "maxunpool_roundtrip_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    exp_y = ort.InferenceSession(model.SerializeToString()).run(None, {"x": x})[0]
    with torch.no_grad():
        y = ConvertModel(model)(torch.from_numpy(x))

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-5, atol=1e-5)
