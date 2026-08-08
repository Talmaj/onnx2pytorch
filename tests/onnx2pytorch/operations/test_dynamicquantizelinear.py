import numpy as np
import onnxruntime as ort
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_dynamic_quantize_linear(x):
    node = helper.make_node(
        "DynamicQuantizeLinear",
        inputs=["x"],
        outputs=["y", "y_scale", "y_zero_point"],
    )
    graph = helper.make_graph(
        [node],
        "dynamicquantizelinear_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [
            helper.make_tensor_value_info("y", TensorProto.UINT8, None),
            helper.make_tensor_value_info("y_scale", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("y_zero_point", TensorProto.UINT8, None),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])

    exp = ort.InferenceSession(model.SerializeToString()).run(None, {"x": x})
    with torch.no_grad():
        out = ConvertModel(model)(torch.from_numpy(x))

    np.testing.assert_array_equal(out[0].numpy(), exp[0])
    np.testing.assert_allclose(out[1].numpy(), exp[1], rtol=1e-6, atol=1e-8)
    np.testing.assert_array_equal(out[2].numpy(), exp[2])
    assert out[0].dtype == torch.uint8
    assert out[2].dtype == torch.uint8


def test_dynamic_quantize_linear_positive():
    np.random.seed(0)
    check_dynamic_quantize_linear(np.random.rand(3, 4).astype(np.float32) * 5)


def test_dynamic_quantize_linear_negative():
    np.random.seed(1)
    check_dynamic_quantize_linear(-np.random.rand(2, 5).astype(np.float32) * 3)


def test_dynamic_quantize_linear_mixed():
    np.random.seed(2)
    check_dynamic_quantize_linear(np.random.randn(2, 3, 4).astype(np.float32) * 10)


def test_dynamic_quantize_linear_1d():
    x = np.array([0.0, 2.0, -3.0, -2.5, 1.34, 0.5], dtype=np.float32)
    check_dynamic_quantize_linear(x)
