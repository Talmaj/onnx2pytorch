import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

from onnx2pytorch.convert import ConvertModel


def build_model(x, y_scale, y_zero_point=None, opset=21, out_type=None, **attrs):
    inputs = ["x", "y_scale"]
    value_infos = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape)),
        helper.make_tensor_value_info(
            "y_scale", TensorProto.FLOAT, list(y_scale.shape)
        ),
    ]
    feed = {"x": x, "y_scale": y_scale}
    if y_zero_point is not None:
        zero_point_type = helper.np_dtype_to_tensor_dtype(y_zero_point.dtype)
        inputs.append("y_zero_point")
        value_infos.append(
            helper.make_tensor_value_info(
                "y_zero_point", zero_point_type, list(y_zero_point.shape)
            )
        )
        feed["y_zero_point"] = y_zero_point
        out_type = zero_point_type
    elif out_type is None:
        out_type = TensorProto.UINT8

    node = helper.make_node("QuantizeLinear", inputs=inputs, outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "quantizelinear_test",
        value_infos,
        [helper.make_tensor_value_info("y", out_type, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    return model, feed


def check_quantize_linear(
    x, y_scale, y_zero_point=None, opset=21, out_type=None, use_reference=False, **attrs
):
    model, feed = build_model(x, y_scale, y_zero_point, opset, out_type, **attrs)
    if use_reference:
        exp_y = ReferenceEvaluator(model).run(None, feed)[0]
    else:
        exp_y = ort.InferenceSession(model.SerializeToString()).run(None, feed)[0]
    with torch.no_grad():
        y = ConvertModel(model)(*[torch.from_numpy(v) for v in feed.values()])
    assert y.numpy().dtype == exp_y.dtype
    np.testing.assert_array_equal(y.numpy(), exp_y)


def test_quantize_linear_per_tensor_uint8():
    np.random.seed(0)
    x = np.random.randn(2, 3, 4).astype(np.float32) * 10
    check_quantize_linear(
        x, np.array(0.5, dtype=np.float32), np.array(128, dtype=np.uint8), opset=13
    )


def test_quantize_linear_per_tensor_int8():
    np.random.seed(1)
    x = np.random.randn(2, 3, 4).astype(np.float32) * 10
    check_quantize_linear(
        x, np.array(0.25, dtype=np.float32), np.array(-7, dtype=np.int8), opset=13
    )


def test_quantize_linear_no_zero_point():
    np.random.seed(2)
    x = np.random.rand(3, 5).astype(np.float32) * 20
    check_quantize_linear(x, np.array(0.1, dtype=np.float32), opset=13)


def test_quantize_linear_rounds_half_to_even():
    x = np.array([0.5, 1.5, 2.5, 3.5, -0.5, -1.5], dtype=np.float32)
    check_quantize_linear(
        x, np.array(1.0, dtype=np.float32), np.array(0, dtype=np.int8), opset=13
    )


def test_quantize_linear_saturates():
    x = np.array([-1000.0, 1000.0, 0.0], dtype=np.float32)
    check_quantize_linear(
        x, np.array(1.0, dtype=np.float32), np.array(10, dtype=np.uint8), opset=13
    )


def test_quantize_linear_int16():
    np.random.seed(3)
    x = np.random.randn(4, 6).astype(np.float32) * 100
    check_quantize_linear(
        x, np.array(0.01, dtype=np.float32), np.array(300, dtype=np.int16), opset=21
    )


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_quantize_linear_per_axis(axis):
    np.random.seed(4)
    x = np.random.randn(2, 3, 4).astype(np.float32) * 5
    size = x.shape[axis]
    y_scale = (np.random.rand(size).astype(np.float32) + 0.1) / 2
    y_zero_point = np.random.randint(0, 255, size=size).astype(np.uint8)
    check_quantize_linear(x, y_scale, y_zero_point, opset=13, axis=axis)


def test_quantize_linear_output_dtype():
    np.random.seed(5)
    x = np.random.randn(2, 4).astype(np.float32) * 3
    check_quantize_linear(
        x,
        np.array(0.2, dtype=np.float32),
        opset=21,
        out_type=TensorProto.INT8,
        output_dtype=TensorProto.INT8,
    )


def test_quantize_linear_blocked():
    np.random.seed(6)
    x = np.random.randn(6, 4).astype(np.float32) * 5
    y_scale = (np.random.rand(3, 4).astype(np.float32) + 0.1) / 2
    y_zero_point = np.random.randint(0, 255, size=(3, 4)).astype(np.uint8)
    check_quantize_linear(
        x,
        y_scale,
        y_zero_point,
        opset=21,
        use_reference=True,
        axis=0,
        block_size=2,
    )
