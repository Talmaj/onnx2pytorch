import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

from onnx2pytorch.convert import ConvertModel


def build_model(signal, frame_step, window=None, frame_length=None, **attrs):
    inputs = ["signal", "frame_step"]
    value_infos = [
        helper.make_tensor_value_info("signal", TensorProto.FLOAT, list(signal.shape)),
        helper.make_tensor_value_info("frame_step", TensorProto.INT64, []),
    ]
    feed = {"signal": signal, "frame_step": frame_step}
    if window is not None:
        inputs.append("window")
        value_infos.append(
            helper.make_tensor_value_info(
                "window", TensorProto.FLOAT, list(window.shape)
            )
        )
        feed["window"] = window
    if frame_length is not None:
        if window is None:
            inputs.append("")
        inputs.append("frame_length")
        value_infos.append(
            helper.make_tensor_value_info("frame_length", TensorProto.INT64, [])
        )
        feed["frame_length"] = frame_length

    node = helper.make_node("STFT", inputs=inputs, outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "stft_test",
        value_infos,
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model, feed


def check_stft(
    signal, frame_step, window=None, frame_length=None, use_reference=False, **attrs
):
    model, feed = build_model(signal, frame_step, window, frame_length, **attrs)
    if use_reference:
        exp_y = ReferenceEvaluator(model).run(None, feed)[0]
    else:
        exp_y = ort.InferenceSession(model.SerializeToString()).run(None, feed)[0]
    with torch.no_grad():
        y = ConvertModel(model)(*[torch.from_numpy(v) for v in feed.values()])
    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-4, atol=1e-4)


def make_signal(batch, length, channels, seed):
    np.random.seed(seed)
    return np.random.randn(batch, length, channels).astype(np.float32)


def test_stft_real_signal():
    signal = make_signal(2, 64, 1, 0)
    check_stft(
        signal,
        np.array(8, dtype=np.int64),
        frame_length=np.array(16, dtype=np.int64),
    )


def test_stft_with_window():
    signal = make_signal(1, 64, 1, 1)
    window = np.hanning(16).astype(np.float32)
    check_stft(signal, np.array(8, dtype=np.int64), window=window)


def test_stft_complex_signal():
    signal = make_signal(2, 32, 2, 2)
    # onnxruntime returns garbage for complex signals, compare against onnx instead
    check_stft(
        signal,
        np.array(4, dtype=np.int64),
        frame_length=np.array(8, dtype=np.int64),
        use_reference=True,
        onesided=0,
    )


@pytest.mark.parametrize("onesided", [0, 1])
def test_stft_onesided(onesided):
    signal = make_signal(1, 48, 1, 3)
    window = np.hamming(12).astype(np.float32)
    check_stft(signal, np.array(6, dtype=np.int64), window=window, onesided=onesided)


@pytest.mark.parametrize("frame_step", [1, 3, 16])
def test_stft_frame_step(frame_step):
    signal = make_signal(1, 64, 1, 4)
    check_stft(
        signal,
        np.array(frame_step, dtype=np.int64),
        frame_length=np.array(16, dtype=np.int64),
    )


def test_stft_window_and_frame_length():
    signal = make_signal(2, 40, 1, 5)
    window = np.blackman(10).astype(np.float32)
    check_stft(
        signal,
        np.array(5, dtype=np.int64),
        window=window,
        frame_length=np.array(10, dtype=np.int64),
    )
