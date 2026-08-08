import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

from onnx2pytorch.convert import ConvertModel


def build_model(x, x_scale, x_zero_point=None, opset=21, **attrs):
    x_type = helper.np_dtype_to_tensor_dtype(x.dtype)
    scale_type = helper.np_dtype_to_tensor_dtype(x_scale.dtype)
    inputs = ["x", "x_scale"]
    value_infos = [
        helper.make_tensor_value_info("x", x_type, list(x.shape)),
        helper.make_tensor_value_info("x_scale", scale_type, list(x_scale.shape)),
    ]
    feed = {"x": x, "x_scale": x_scale}
    if x_zero_point is not None:
        inputs.append("x_zero_point")
        value_infos.append(
            helper.make_tensor_value_info(
                "x_zero_point",
                helper.np_dtype_to_tensor_dtype(x_zero_point.dtype),
                list(x_zero_point.shape),
            )
        )
        feed["x_zero_point"] = x_zero_point

    node = helper.make_node("DequantizeLinear", inputs=inputs, outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "dequantizelinear_test",
        value_infos,
        [helper.make_tensor_value_info("y", scale_type, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    return model, feed


def check_dequantize_linear(
    x, x_scale, x_zero_point=None, opset=21, use_reference=False, **attrs
):
    model, feed = build_model(x, x_scale, x_zero_point, opset, **attrs)
    if use_reference:
        exp_y = ReferenceEvaluator(model).run(None, feed)[0]
    else:
        exp_y = ort.InferenceSession(model.SerializeToString()).run(None, feed)[0]
    with torch.no_grad():
        y = ConvertModel(model)(*[torch.from_numpy(v) for v in feed.values()])
    assert y.numpy().dtype == exp_y.dtype
    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-6, atol=1e-6)


def test_dequantize_linear_per_tensor_uint8():
    np.random.seed(0)
    x = np.random.randint(0, 255, size=(2, 3, 4)).astype(np.uint8)
    check_dequantize_linear(
        x,
        np.array(0.5, dtype=np.float32),
        np.array(128, dtype=np.uint8),
        opset=13,
    )


def test_dequantize_linear_per_tensor_int8():
    np.random.seed(1)
    x = np.random.randint(-128, 127, size=(3, 5)).astype(np.int8)
    check_dequantize_linear(
        x, np.array(0.02, dtype=np.float32), np.array(-3, dtype=np.int8), opset=13
    )


def test_dequantize_linear_no_zero_point():
    np.random.seed(2)
    x = np.random.randint(0, 255, size=(4, 4)).astype(np.uint8)
    check_dequantize_linear(x, np.array(0.125, dtype=np.float32), opset=13)


def test_dequantize_linear_int32():
    np.random.seed(3)
    x = np.random.randint(-10000, 10000, size=(2, 6)).astype(np.int32)
    check_dequantize_linear(x, np.array(0.001, dtype=np.float32), opset=13)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_dequantize_linear_per_axis(axis):
    np.random.seed(4)
    x = np.random.randint(0, 255, size=(2, 3, 4)).astype(np.uint8)
    size = x.shape[axis]
    x_scale = (np.random.rand(size).astype(np.float32) + 0.1) / 2
    x_zero_point = np.random.randint(0, 255, size=size).astype(np.uint8)
    check_dequantize_linear(x, x_scale, x_zero_point, opset=13, axis=axis)


def test_dequantize_linear_blocked():
    np.random.seed(5)
    x = np.random.randint(0, 255, size=(6, 4)).astype(np.uint8)
    x_scale = (np.random.rand(3, 4).astype(np.float32) + 0.1) / 2
    x_zero_point = np.random.randint(0, 255, size=(3, 4)).astype(np.uint8)
    check_dequantize_linear(
        x,
        x_scale,
        x_zero_point,
        opset=21,
        use_reference=True,
        axis=0,
        block_size=2,
    )


def test_dequantize_linear_float16_scale():
    np.random.seed(6)
    x = np.random.randint(0, 255, size=(3, 4)).astype(np.uint8)
    check_dequantize_linear(
        x,
        np.array(0.5, dtype=np.float16),
        np.array(100, dtype=np.uint8),
        opset=19,
    )
