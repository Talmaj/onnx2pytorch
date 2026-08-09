import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel

INPUT_NAMES = [
    "num_mel_bins",
    "dft_length",
    "sample_rate",
    "lower_edge_hertz",
    "upper_edge_hertz",
]


def check_mel_weight_matrix(
    num_mel_bins,
    dft_length,
    sample_rate,
    lower_edge_hertz,
    upper_edge_hertz,
    out_type=TensorProto.FLOAT,
    **attrs
):
    feed = {
        "num_mel_bins": np.array(num_mel_bins, dtype=np.int64),
        "dft_length": np.array(dft_length, dtype=np.int64),
        "sample_rate": np.array(sample_rate, dtype=np.int64),
        "lower_edge_hertz": np.array(lower_edge_hertz, dtype=np.float32),
        "upper_edge_hertz": np.array(upper_edge_hertz, dtype=np.float32),
    }
    value_infos = [
        helper.make_tensor_value_info(
            name, helper.np_dtype_to_tensor_dtype(value.dtype), []
        )
        for name, value in feed.items()
    ]

    node = helper.make_node(
        "MelWeightMatrix", inputs=INPUT_NAMES, outputs=["y"], **attrs
    )
    graph = helper.make_graph(
        [node],
        "melweightmatrix_test",
        value_infos,
        [helper.make_tensor_value_info("y", out_type, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    exp_y = ort.InferenceSession(model.SerializeToString()).run(None, feed)[0]
    with torch.no_grad():
        y = ConvertModel(model)(*[torch.from_numpy(v) for v in feed.values()])
    assert y.numpy().dtype == exp_y.dtype
    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-5, atol=1e-6)


def test_mel_weight_matrix():
    check_mel_weight_matrix(8, 16, 8192, 0.0, 8192 / 2)


def test_mel_weight_matrix_speech():
    check_mel_weight_matrix(20, 512, 16000, 20.0, 7600.0)


@pytest.mark.parametrize("num_mel_bins", [1, 4, 32])
def test_mel_weight_matrix_bins(num_mel_bins):
    check_mel_weight_matrix(num_mel_bins, 256, 16000, 0.0, 8000.0)


def test_mel_weight_matrix_narrow_band():
    check_mel_weight_matrix(6, 64, 16000, 1000.0, 2000.0)


def test_mel_weight_matrix_double():
    check_mel_weight_matrix(
        8,
        128,
        16000,
        0.0,
        8000.0,
        out_type=TensorProto.DOUBLE,
        output_datatype=TensorProto.DOUBLE,
    )
