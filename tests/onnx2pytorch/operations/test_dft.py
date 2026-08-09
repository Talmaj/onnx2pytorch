import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def build_model(x, dft_length=None, axis=None, opset=20, **attrs):
    inputs = ["x"]
    value_infos = [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))]
    feed = {"x": x}
    if dft_length is not None:
        inputs.append("dft_length")
        value_infos.append(
            helper.make_tensor_value_info("dft_length", TensorProto.INT64, [])
        )
        feed["dft_length"] = dft_length
    if axis is not None:
        if dft_length is None:
            inputs.append("")
        inputs.append("axis")
        value_infos.append(helper.make_tensor_value_info("axis", TensorProto.INT64, []))
        feed["axis"] = axis

    node = helper.make_node("DFT", inputs=inputs, outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "dft_test",
        value_infos,
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    return model, feed


def check_dft(x, dft_length=None, axis=None, opset=20, **attrs):
    model, feed = build_model(x, dft_length, axis, opset, **attrs)
    exp_y = ort.InferenceSession(model.SerializeToString()).run(None, feed)[0]
    with torch.no_grad():
        y = ConvertModel(model)(*[torch.from_numpy(v) for v in feed.values()])
    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-4, atol=1e-4)


def test_dft_real_signal():
    np.random.seed(0)
    x = np.random.randn(2, 8, 1).astype(np.float32)
    check_dft(x, axis=np.array(1, dtype=np.int64))


def test_dft_complex_signal():
    np.random.seed(1)
    x = np.random.randn(2, 8, 2).astype(np.float32)
    check_dft(x, axis=np.array(1, dtype=np.int64))


def test_dft_onesided():
    np.random.seed(2)
    x = np.random.randn(3, 16, 1).astype(np.float32)
    check_dft(x, axis=np.array(1, dtype=np.int64), onesided=1)


def test_dft_inverse():
    np.random.seed(3)
    x = np.random.randn(2, 8, 2).astype(np.float32)
    check_dft(x, axis=np.array(1, dtype=np.int64), inverse=1)


def test_dft_roundtrip():
    np.random.seed(4)
    x = np.random.randn(2, 8, 2).astype(np.float32)
    axis = np.array(1, dtype=np.int64)

    forward_model, _ = build_model(x, axis=axis)
    inverse_model, _ = build_model(x, axis=axis, inverse=1)
    with torch.no_grad():
        y = ConvertModel(forward_model)(torch.from_numpy(x), torch.from_numpy(axis))
        restored = ConvertModel(inverse_model)(y, torch.from_numpy(axis))

    np.testing.assert_allclose(restored.numpy(), x, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dft_length", [4, 8, 16])
def test_dft_dft_length(dft_length):
    np.random.seed(5)
    x = np.random.randn(2, 8, 1).astype(np.float32)
    check_dft(
        x,
        dft_length=np.array(dft_length, dtype=np.int64),
        axis=np.array(1, dtype=np.int64),
    )


def test_dft_multidimensional_signal():
    np.random.seed(6)
    x = np.random.randn(2, 4, 6, 1).astype(np.float32)
    check_dft(x, axis=np.array(2, dtype=np.int64))


def test_dft_negative_axis():
    np.random.seed(7)
    x = np.random.randn(2, 4, 6, 2).astype(np.float32)
    check_dft(x, axis=np.array(-2, dtype=np.int64))


def test_dft_axis_attribute_opset17():
    np.random.seed(8)
    x = np.random.randn(2, 8, 1).astype(np.float32)
    check_dft(x, opset=17, axis=None)


def test_dft_axis_attribute_opset17_explicit():
    np.random.seed(9)
    x = np.random.randn(2, 5, 6, 1).astype(np.float32)
    model, feed = build_model(x, opset=17)
    model.graph.node[0].attribute.append(helper.make_attribute("axis", 2))
    exp_y = ort.InferenceSession(model.SerializeToString()).run(None, feed)[0]
    with torch.no_grad():
        y = ConvertModel(model)(*[torch.from_numpy(v) for v in feed.values()])
    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-4, atol=1e-4)
