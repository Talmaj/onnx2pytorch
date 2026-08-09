import numpy as np
import onnxruntime as ort
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel

INPUT_NAMES = [
    "x",
    "x_scale",
    "x_zero_point",
    "w",
    "w_scale",
    "w_zero_point",
    "y_scale",
    "y_zero_point",
    "b",
]


def build_model(arrays, **attrs):
    feed = {
        name: value for name, value in zip(INPUT_NAMES, arrays) if value is not None
    }
    value_infos = [
        helper.make_tensor_value_info(
            name, helper.np_dtype_to_tensor_dtype(value.dtype), list(value.shape)
        )
        for name, value in feed.items()
    ]
    out_type = helper.np_dtype_to_tensor_dtype(feed["y_zero_point"].dtype)

    node = helper.make_node(
        "QLinearConv", inputs=list(feed.keys()), outputs=["y"], **attrs
    )
    graph = helper.make_graph(
        [node],
        "qlinearconv_test",
        value_infos,
        [helper.make_tensor_value_info("y", out_type, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    return model, feed


def check_qlinear_conv(arrays, **attrs):
    model, feed = build_model(arrays, **attrs)
    exp_y = ort.InferenceSession(model.SerializeToString()).run(None, feed)[0]
    with torch.no_grad():
        y = ConvertModel(model)(*[torch.from_numpy(v) for v in feed.values()])
    assert y.numpy().dtype == exp_y.dtype
    np.testing.assert_array_equal(y.numpy(), exp_y)


def make_inputs(x_shape, w_shape, seed, w_scale=None, w_zero_point=None, bias=None):
    np.random.seed(seed)
    return [
        np.random.randint(0, 255, size=x_shape).astype(np.uint8),
        np.array(0.02, dtype=np.float32),
        np.array(120, dtype=np.uint8),
        np.random.randint(0, 255, size=w_shape).astype(np.uint8),
        np.array(0.03, dtype=np.float32) if w_scale is None else w_scale,
        np.array(130, dtype=np.uint8) if w_zero_point is None else w_zero_point,
        np.array(0.5, dtype=np.float32),
        np.array(100, dtype=np.uint8),
        bias,
    ]


def test_qlinear_conv():
    check_qlinear_conv(make_inputs((1, 1, 5, 5), (1, 1, 3, 3), 0), kernel_shape=[3, 3])


def test_qlinear_conv_multi_channel():
    check_qlinear_conv(make_inputs((2, 3, 6, 6), (4, 3, 3, 3), 1), kernel_shape=[3, 3])


def test_qlinear_conv_bias():
    bias = np.array([1000, -2000, 500, 0], dtype=np.int32)
    check_qlinear_conv(
        make_inputs((1, 3, 6, 6), (4, 3, 3, 3), 2, bias=bias), kernel_shape=[3, 3]
    )


def test_qlinear_conv_per_channel_weight_scale():
    w_scale = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32)
    w_zero_point = np.full(4, 130, dtype=np.uint8)
    check_qlinear_conv(
        make_inputs(
            (1, 2, 5, 5),
            (4, 2, 3, 3),
            3,
            w_scale=w_scale,
            w_zero_point=w_zero_point,
        ),
        kernel_shape=[3, 3],
    )


def test_qlinear_conv_per_channel_weight_zero_point():
    w_scale = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32)
    w_zero_point = np.array([120, 125, 130, 135], dtype=np.uint8)
    check_qlinear_conv(
        make_inputs(
            (1, 2, 5, 5),
            (4, 2, 3, 3),
            8,
            w_scale=w_scale,
            w_zero_point=w_zero_point,
        ),
        kernel_shape=[3, 3],
    )


def test_qlinear_conv_pads_strides_dilations():
    check_qlinear_conv(
        make_inputs((1, 1, 8, 8), (2, 1, 3, 3), 4),
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[2, 2],
        dilations=[2, 2],
    )


def test_qlinear_conv_groups():
    check_qlinear_conv(
        make_inputs((1, 4, 5, 5), (4, 2, 3, 3), 5), kernel_shape=[3, 3], group=2
    )


def test_qlinear_conv_auto_pad():
    check_qlinear_conv(
        make_inputs((1, 1, 5, 5), (1, 1, 3, 3), 6),
        kernel_shape=[3, 3],
        strides=[2, 2],
        auto_pad="SAME_UPPER",
    )


def test_qlinear_conv_int8():
    np.random.seed(7)
    arrays = [
        np.random.randint(-128, 127, size=(1, 2, 5, 5)).astype(np.int8),
        np.array(0.01, dtype=np.float32),
        np.array(-5, dtype=np.int8),
        np.random.randint(-128, 127, size=(3, 2, 3, 3)).astype(np.int8),
        np.array(0.02, dtype=np.float32),
        np.array(3, dtype=np.int8),
        np.array(0.2, dtype=np.float32),
        np.array(-10, dtype=np.int8),
        None,
    ]
    check_qlinear_conv(arrays, kernel_shape=[3, 3])
