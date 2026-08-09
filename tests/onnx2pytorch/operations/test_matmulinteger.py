import numpy as np
import onnxruntime as ort
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def build_model(a, b, a_zero_point=None, b_zero_point=None):
    inputs = ["a", "b"]
    value_infos = [
        helper.make_tensor_value_info(
            "a", helper.np_dtype_to_tensor_dtype(a.dtype), list(a.shape)
        ),
        helper.make_tensor_value_info(
            "b", helper.np_dtype_to_tensor_dtype(b.dtype), list(b.shape)
        ),
    ]
    feed = {"a": a, "b": b}
    for name, zero_point in (
        ("a_zero_point", a_zero_point),
        ("b_zero_point", b_zero_point),
    ):
        if zero_point is None:
            if name == "a_zero_point" and b_zero_point is not None:
                inputs.append("")
            continue
        inputs.append(name)
        value_infos.append(
            helper.make_tensor_value_info(
                name,
                helper.np_dtype_to_tensor_dtype(zero_point.dtype),
                list(zero_point.shape),
            )
        )
        feed[name] = zero_point

    node = helper.make_node("MatMulInteger", inputs=inputs, outputs=["y"])
    graph = helper.make_graph(
        [node],
        "matmulinteger_test",
        value_infos,
        [helper.make_tensor_value_info("y", TensorProto.INT32, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    return model, feed


def check_matmul_integer(a, b, a_zero_point=None, b_zero_point=None):
    model, feed = build_model(a, b, a_zero_point, b_zero_point)
    exp_y = ort.InferenceSession(model.SerializeToString()).run(None, feed)[0]
    with torch.no_grad():
        y = ConvertModel(model)(*[torch.from_numpy(v) for v in feed.values()])
    assert y.dtype == torch.int32
    np.testing.assert_array_equal(y.numpy(), exp_y)


def test_matmul_integer_uint8():
    np.random.seed(0)
    a = np.random.randint(0, 255, size=(4, 3)).astype(np.uint8)
    b = np.random.randint(0, 255, size=(3, 2)).astype(np.uint8)
    check_matmul_integer(a, b)


def test_matmul_integer_int8():
    np.random.seed(1)
    a = np.random.randint(-128, 127, size=(5, 6)).astype(np.int8)
    b = np.random.randint(-128, 127, size=(6, 4)).astype(np.int8)
    check_matmul_integer(a, b)


def test_matmul_integer_scalar_zero_points():
    np.random.seed(2)
    a = np.random.randint(0, 255, size=(4, 3)).astype(np.uint8)
    b = np.random.randint(0, 255, size=(3, 2)).astype(np.uint8)
    check_matmul_integer(
        a, b, np.array(120, dtype=np.uint8), np.array(130, dtype=np.uint8)
    )


def test_matmul_integer_per_row_zero_point():
    np.random.seed(3)
    a = np.random.randint(0, 255, size=(4, 3)).astype(np.uint8)
    b = np.random.randint(0, 255, size=(3, 2)).astype(np.uint8)
    a_zero_point = np.random.randint(0, 255, size=(4,)).astype(np.uint8)

    # onnxruntime rejects per-row zero points, so the oracle is one onnxruntime
    # run per row with that row's scalar zero point
    rows = []
    for row, zero_point in enumerate(a_zero_point):
        row_model, row_feed = build_model(
            a[row : row + 1], b, np.array(zero_point, dtype=np.uint8)
        )
        rows.append(
            ort.InferenceSession(row_model.SerializeToString()).run(None, row_feed)[0]
        )
    exp_y = np.concatenate(rows, axis=0)

    model, feed = build_model(a, b, a_zero_point)
    with torch.no_grad():
        y = ConvertModel(model)(*[torch.from_numpy(v) for v in feed.values()])
    np.testing.assert_array_equal(y.numpy(), exp_y)


def test_matmul_integer_per_column_zero_point():
    np.random.seed(4)
    a = np.random.randint(0, 255, size=(4, 3)).astype(np.uint8)
    b = np.random.randint(0, 255, size=(3, 2)).astype(np.uint8)
    b_zero_point = np.random.randint(0, 255, size=(2,)).astype(np.uint8)
    check_matmul_integer(a, b, None, b_zero_point)


def test_matmul_integer_batched():
    np.random.seed(5)
    a = np.random.randint(0, 255, size=(2, 4, 3)).astype(np.uint8)
    b = np.random.randint(0, 255, size=(2, 3, 5)).astype(np.uint8)
    check_matmul_integer(a, b, np.array(10, dtype=np.uint8))


def test_matmul_integer_mixed_signedness():
    np.random.seed(6)
    a = np.random.randint(0, 255, size=(3, 4)).astype(np.uint8)
    b = np.random.randint(-128, 127, size=(4, 3)).astype(np.int8)
    check_matmul_integer(
        a, b, np.array(100, dtype=np.uint8), np.array(-5, dtype=np.int8)
    )
