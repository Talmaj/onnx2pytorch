import numpy as np
import onnxruntime as ort
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_binary_op(op_type, x, y, in_type, out_type, opset_version=13):
    node = helper.make_node(op_type, inputs=["x", "y"], outputs=["z"])
    graph = helper.make_graph(
        [node],
        "{}_test".format(op_type.lower()),
        [
            helper.make_tensor_value_info("x", in_type, list(x.shape)),
            helper.make_tensor_value_info("y", in_type, list(y.shape)),
        ],
        [helper.make_tensor_value_info("z", out_type, None)],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset_version)]
    )

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_z = ort_session.run(None, {"x": x, "y": y})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        z = o2p_model(torch.from_numpy(x), torch.from_numpy(y))

    np.testing.assert_array_equal(z.numpy(), exp_z)


def test_less_or_equal():
    np.random.seed(0)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.randn(3, 4, 5).astype(np.float32)
    check_binary_op("LessOrEqual", x, y, TensorProto.FLOAT, TensorProto.BOOL)


def test_less_or_equal_broadcast():
    np.random.seed(0)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.randn(5).astype(np.float32)
    check_binary_op("LessOrEqual", x, y, TensorProto.FLOAT, TensorProto.BOOL)


def test_bitwise_and():
    np.random.seed(0)
    x = np.random.randint(-100, 100, (3, 4, 5)).astype(np.int32)
    y = np.random.randint(-100, 100, (3, 4, 5)).astype(np.int32)
    check_binary_op(
        "BitwiseAnd", x, y, TensorProto.INT32, TensorProto.INT32, opset_version=18
    )


def test_bitwise_and_uint8_broadcast():
    np.random.seed(0)
    x = np.random.randint(0, 255, (3, 4, 5)).astype(np.uint8)
    y = np.random.randint(0, 255, (5,)).astype(np.uint8)
    check_binary_op(
        "BitwiseAnd", x, y, TensorProto.UINT8, TensorProto.UINT8, opset_version=18
    )


def test_xor():
    np.random.seed(0)
    x = np.random.randn(3, 4, 5) > 0
    y = np.random.randn(3, 4, 5) > 0
    check_binary_op("Xor", x, y, TensorProto.BOOL, TensorProto.BOOL)


def test_xor_broadcast():
    np.random.seed(0)
    x = np.random.randn(3, 4, 5) > 0
    y = np.random.randn(4, 5) > 0
    check_binary_op("Xor", x, y, TensorProto.BOOL, TensorProto.BOOL)
