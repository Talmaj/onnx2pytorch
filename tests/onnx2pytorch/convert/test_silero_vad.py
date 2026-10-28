"""Silero VAD: Constant weights live inside If branches (onnx2torch #239)."""

import os
import urllib.request

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from onnx2pytorch.convert import ConvertModel

URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
CACHE = os.path.join(os.path.dirname(__file__), "_cache", "silero_vad.onnx")


def _model_path():
    if os.path.exists(CACHE) and os.path.getsize(CACHE) > 0:
        return CACHE
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    urllib.request.urlretrieve(URL, CACHE)
    return CACHE


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize(
    "sample_rate, frames",
    [(16000, 512), (8000, 256)],
)
def test_silero_vad_matches_onnxruntime(sample_rate, frames):
    path = _model_path()
    rng = np.random.default_rng(0)
    feeds = {
        "input": rng.standard_normal((1, frames)).astype(np.float32),
        "state": np.zeros((2, 1, 128), dtype=np.float32),
        "sr": np.array(sample_rate, dtype=np.int64),
    }
    expected = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(
        None, feeds
    )
    converted = ConvertModel(onnx.load(path))
    converted.eval()
    with torch.no_grad():
        actual = converted(
            torch.from_numpy(feeds["input"]),
            torch.from_numpy(feeds["state"]),
            torch.from_numpy(feeds["sr"]),
        )
    diffs = []
    for got, want in zip(actual, expected):
        got = got.detach().cpu().numpy()
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
        diffs.append(
            float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64))))
        )
    # Printed by pytest on failure via the assertion; keep the values on the test.
    assert diffs
