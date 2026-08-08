import numpy as np
import onnxruntime as ort
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_unary_op(op_type, x, opset_version=13):
    node = helper.make_node(op_type, inputs=["x"], outputs=["y"])
    graph = helper.make_graph(
        [node],
        "{}_test".format(op_type.lower()),
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(x.shape))],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset_version)]
    )

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_y = ort_session.run(None, {"x": x})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(torch.from_numpy(x))

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-6, atol=1e-6)


def test_acos():
    check_unary_op("Acos", np.array([-0.5, 0.0, 0.5], dtype=np.float32))
    np.random.seed(0)
    check_unary_op("Acos", np.random.uniform(-1, 1, (3, 4, 5)).astype(np.float32))


def test_acosh():
    check_unary_op("Acosh", np.array([10.0, np.e, 1.0], dtype=np.float32))
    np.random.seed(0)
    check_unary_op("Acosh", np.random.uniform(1, 10, (3, 4, 5)).astype(np.float32))


def test_asin():
    check_unary_op("Asin", np.array([-0.5, 0.0, 0.5], dtype=np.float32))
    np.random.seed(0)
    check_unary_op("Asin", np.random.uniform(-1, 1, (3, 4, 5)).astype(np.float32))


def test_asinh():
    check_unary_op("Asinh", np.array([-1.0, 0.0, 1.0], dtype=np.float32))
    np.random.seed(0)
    check_unary_op("Asinh", np.random.randn(3, 4, 5).astype(np.float32))


def test_atan():
    check_unary_op("Atan", np.array([-1.0, 0.0, 1.0], dtype=np.float32))
    np.random.seed(0)
    check_unary_op("Atan", np.random.randn(3, 4, 5).astype(np.float32))


def test_atanh():
    check_unary_op("Atanh", np.array([-0.5, 0.0, 0.5], dtype=np.float32))
    np.random.seed(0)
    check_unary_op("Atanh", np.random.uniform(-0.9, 0.9, (3, 4, 5)).astype(np.float32))


def test_cos():
    check_unary_op("Cos", np.array([-1.0, 0.0, 1.0], dtype=np.float32))
    np.random.seed(0)
    check_unary_op("Cos", np.random.randn(3, 4, 5).astype(np.float32))


def test_cosh():
    check_unary_op("Cosh", np.array([-1.0, 0.0, 1.0], dtype=np.float32))
    np.random.seed(0)
    check_unary_op("Cosh", np.random.randn(3, 4, 5).astype(np.float32))


def test_sin():
    check_unary_op("Sin", np.array([-1.0, 0.0, 1.0], dtype=np.float32))
    np.random.seed(0)
    check_unary_op("Sin", np.random.randn(3, 4, 5).astype(np.float32))


def test_sinh():
    check_unary_op("Sinh", np.array([-1.0, 0.0, 1.0], dtype=np.float32))
    np.random.seed(0)
    check_unary_op("Sinh", np.random.randn(3, 4, 5).astype(np.float32))
