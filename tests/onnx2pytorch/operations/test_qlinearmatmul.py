import numpy as np
import onnxruntime as ort
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel

INPUT_NAMES = [
    "a",
    "a_scale",
    "a_zero_point",
    "b",
    "b_scale",
    "b_zero_point",
    "y_scale",
    "y_zero_point",
]


def build_model(*arrays):
    feed = dict(zip(INPUT_NAMES, arrays))
    value_infos = [
        helper.make_tensor_value_info(
            name, helper.np_dtype_to_tensor_dtype(value.dtype), list(value.shape)
        )
        for name, value in feed.items()
    ]
    out_type = helper.np_dtype_to_tensor_dtype(feed["y_zero_point"].dtype)

    node = helper.make_node("QLinearMatMul", inputs=INPUT_NAMES, outputs=["y"])
    graph = helper.make_graph(
        [node],
        "qlinearmatmul_test",
        value_infos,
        [helper.make_tensor_value_info("y", out_type, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    return model, feed


def check_qlinear_matmul(*arrays):
    model, feed = build_model(*arrays)
    exp_y = ort.InferenceSession(model.SerializeToString()).run(None, feed)[0]
    with torch.no_grad():
        y = ConvertModel(model)(*[torch.from_numpy(v) for v in feed.values()])
    assert y.numpy().dtype == exp_y.dtype
    np.testing.assert_array_equal(y.numpy(), exp_y)


def test_qlinear_matmul_uint8():
    np.random.seed(0)
    check_qlinear_matmul(
        np.random.randint(0, 255, size=(4, 3)).astype(np.uint8),
        np.array(0.02, dtype=np.float32),
        np.array(120, dtype=np.uint8),
        np.random.randint(0, 255, size=(3, 5)).astype(np.uint8),
        np.array(0.03, dtype=np.float32),
        np.array(130, dtype=np.uint8),
        np.array(0.05, dtype=np.float32),
        np.array(100, dtype=np.uint8),
    )


def test_qlinear_matmul_int8():
    np.random.seed(1)
    check_qlinear_matmul(
        np.random.randint(-128, 127, size=(6, 4)).astype(np.int8),
        np.array(0.01, dtype=np.float32),
        np.array(-3, dtype=np.int8),
        np.random.randint(-128, 127, size=(4, 2)).astype(np.int8),
        np.array(0.015, dtype=np.float32),
        np.array(5, dtype=np.int8),
        np.array(0.04, dtype=np.float32),
        np.array(-10, dtype=np.int8),
    )


def test_qlinear_matmul_batched():
    np.random.seed(2)
    check_qlinear_matmul(
        np.random.randint(0, 255, size=(2, 4, 3)).astype(np.uint8),
        np.array(0.02, dtype=np.float32),
        np.array(120, dtype=np.uint8),
        np.random.randint(0, 255, size=(2, 3, 5)).astype(np.uint8),
        np.array(0.03, dtype=np.float32),
        np.array(130, dtype=np.uint8),
        np.array(0.1, dtype=np.float32),
        np.array(64, dtype=np.uint8),
    )


def test_qlinear_matmul_saturates():
    np.random.seed(3)
    check_qlinear_matmul(
        np.random.randint(0, 255, size=(3, 8)).astype(np.uint8),
        np.array(0.5, dtype=np.float32),
        np.array(0, dtype=np.uint8),
        np.random.randint(0, 255, size=(8, 3)).astype(np.uint8),
        np.array(0.5, dtype=np.float32),
        np.array(0, dtype=np.uint8),
        np.array(0.01, dtype=np.float32),
        np.array(0, dtype=np.uint8),
    )


def test_qlinear_matmul_per_axis_scales():
    np.random.seed(5)
    a = np.random.randint(0, 255, size=(4, 3)).astype(np.uint8)
    a_scale = np.array([0.02, 0.03, 0.01, 0.04], dtype=np.float32)
    a_zero_point = np.array(120, dtype=np.uint8)
    b = np.random.randint(0, 255, size=(3, 5)).astype(np.uint8)
    b_scale = np.array([0.03, 0.02, 0.05, 0.01, 0.04], dtype=np.float32)
    b_zero_point = np.array(130, dtype=np.uint8)
    y_scale = np.array([0.05, 0.04, 0.03, 0.06], dtype=np.float32)
    y_zero_point = np.array(100, dtype=np.uint8)

    # neither onnxruntime nor the onnx reference support per-axis scales
    acc = (a.astype(np.int32) - np.int32(a_zero_point)) @ (
        b.astype(np.int32) - np.int32(b_zero_point)
    )
    scaled = acc * a_scale[:, None].astype(np.float64) * b_scale.astype(np.float64)
    scaled = scaled / y_scale[:, None].astype(np.float64)
    exp_y = np.clip(np.round(scaled) + np.float64(y_zero_point), 0, 255).astype(
        np.uint8
    )

    model, feed = build_model(
        a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point
    )
    with torch.no_grad():
        y = ConvertModel(model)(*[torch.from_numpy(v) for v in feed.values()])
    np.testing.assert_array_equal(y.numpy(), exp_y)


def test_qlinear_matmul_zero_point_shifts_output():
    np.random.seed(4)
    check_qlinear_matmul(
        np.random.randint(0, 60, size=(3, 3)).astype(np.uint8),
        np.array(0.1, dtype=np.float32),
        np.array(30, dtype=np.uint8),
        np.random.randint(0, 60, size=(3, 3)).astype(np.uint8),
        np.array(0.1, dtype=np.float32),
        np.array(30, dtype=np.uint8),
        np.array(0.2, dtype=np.float32),
        np.array(200, dtype=np.uint8),
    )
