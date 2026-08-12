import numpy as np
import onnxruntime as ort
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel
from tests.onnx2pytorch.differential import run_reference


def build_model(x, w, x_zero_point=None, w_zero_point=None, **attrs):
    inputs = ["x", "w"]
    value_infos = [
        helper.make_tensor_value_info("x", TensorProto.UINT8, list(x.shape)),
        helper.make_tensor_value_info("w", TensorProto.UINT8, list(w.shape)),
    ]
    feed = {"x": x, "w": w}
    if x_zero_point is not None:
        inputs.append("x_zero_point")
        value_infos.append(
            helper.make_tensor_value_info(
                "x_zero_point", TensorProto.UINT8, list(x_zero_point.shape)
            )
        )
        feed["x_zero_point"] = x_zero_point
    if w_zero_point is not None:
        if x_zero_point is None:
            inputs.append("")
        inputs.append("w_zero_point")
        value_infos.append(
            helper.make_tensor_value_info(
                "w_zero_point", TensorProto.UINT8, list(w_zero_point.shape)
            )
        )
        feed["w_zero_point"] = w_zero_point

    node = helper.make_node("ConvInteger", inputs=inputs, outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "convinteger_test",
        value_infos,
        [helper.make_tensor_value_info("y", TensorProto.INT32, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model, feed


def check_conv_integer(x, w, x_zero_point=None, w_zero_point=None, **attrs):
    model, feed = build_model(x, w, x_zero_point, w_zero_point, **attrs)
    exp_y = ort.InferenceSession(model.SerializeToString()).run(None, feed)[0]
    with torch.no_grad():
        y = ConvertModel(model)(*[torch.from_numpy(v) for v in feed.values()])
    np.testing.assert_array_equal(y.numpy(), exp_y)


def test_conv_integer():
    np.random.seed(0)
    x = np.random.randint(0, 255, size=(1, 1, 5, 5), dtype=np.uint8)
    w = np.random.randint(0, 255, size=(1, 1, 3, 3), dtype=np.uint8)
    check_conv_integer(x, w, kernel_shape=[3, 3])


def test_conv_integer_zero_points():
    np.random.seed(0)
    x = np.random.randint(0, 255, size=(2, 2, 6, 6), dtype=np.uint8)
    w = np.random.randint(0, 255, size=(3, 2, 3, 3), dtype=np.uint8)
    x_zero_point = np.array(120, dtype=np.uint8)
    w_zero_point = np.array(100, dtype=np.uint8)
    check_conv_integer(x, w, x_zero_point, w_zero_point, kernel_shape=[3, 3])


def test_conv_integer_per_channel_weight_zero_point():
    """onnxruntime only supports per-tensor zero points, the reference does not."""
    np.random.seed(0)
    x = np.random.randint(0, 255, size=(1, 2, 5, 5), dtype=np.uint8)
    w = np.random.randint(0, 255, size=(3, 2, 3, 3), dtype=np.uint8)
    x_zero_point = np.array(120, dtype=np.uint8)
    w_zero_point = np.array([100, 110, 130], dtype=np.uint8)

    model, feed = build_model(x, w, x_zero_point, w_zero_point, kernel_shape=[3, 3])
    exp_y = run_reference(model, feed)[0]
    with torch.no_grad():
        y = ConvertModel(model)(*[torch.from_numpy(v) for v in feed.values()])
    np.testing.assert_array_equal(y.numpy(), exp_y)


def test_conv_integer_pads_strides_dilations():
    np.random.seed(0)
    x = np.random.randint(0, 255, size=(1, 1, 8, 8), dtype=np.uint8)
    w = np.random.randint(0, 255, size=(2, 1, 3, 3), dtype=np.uint8)
    check_conv_integer(
        x,
        w,
        np.array(50, dtype=np.uint8),
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[2, 2],
        dilations=[2, 2],
    )


def test_conv_integer_groups():
    np.random.seed(0)
    x = np.random.randint(0, 255, size=(1, 4, 5, 5), dtype=np.uint8)
    w = np.random.randint(0, 255, size=(4, 2, 3, 3), dtype=np.uint8)
    check_conv_integer(x, w, kernel_shape=[3, 3], group=2)


def test_conv_integer_auto_pad():
    np.random.seed(0)
    x = np.random.randint(0, 255, size=(1, 1, 5, 5), dtype=np.uint8)
    w = np.random.randint(0, 255, size=(1, 1, 3, 3), dtype=np.uint8)
    check_conv_integer(
        x,
        w,
        np.array(10, dtype=np.uint8),
        kernel_shape=[3, 3],
        strides=[2, 2],
        auto_pad="SAME_UPPER",
    )
