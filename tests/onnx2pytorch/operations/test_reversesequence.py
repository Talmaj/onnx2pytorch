import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_reverse_sequence(x, sequence_lens, **attrs):
    node = helper.make_node(
        "ReverseSequence", inputs=["x", "sequence_lens"], outputs=["y"], **attrs
    )
    graph = helper.make_graph(
        [node],
        "reversesequence_test",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape)),
            helper.make_tensor_value_info(
                "sequence_lens", TensorProto.INT64, list(sequence_lens.shape)
            ),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    exp_y = ort.InferenceSession(model.SerializeToString()).run(
        None, {"x": x, "sequence_lens": sequence_lens}
    )[0]

    with torch.no_grad():
        y = ConvertModel(model)(torch.from_numpy(x), torch.from_numpy(sequence_lens))

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-5, atol=1e-5)


def test_reverse_sequence_time_major():
    x = np.arange(16).reshape(4, 4).astype(np.float32)
    sequence_lens = np.array([4, 3, 2, 1], dtype=np.int64)
    check_reverse_sequence(x, sequence_lens, batch_axis=1, time_axis=0)


def test_reverse_sequence_batch_major():
    x = np.arange(16).reshape(4, 4).astype(np.float32)
    sequence_lens = np.array([1, 2, 3, 4], dtype=np.int64)
    check_reverse_sequence(x, sequence_lens, batch_axis=0, time_axis=1)


def test_reverse_sequence_defaults():
    np.random.seed(0)
    x = np.random.randn(5, 3).astype(np.float32)
    sequence_lens = np.array([5, 2, 3], dtype=np.int64)
    check_reverse_sequence(x, sequence_lens)


@pytest.mark.parametrize("batch_axis, time_axis", [(0, 1), (1, 0)])
def test_reverse_sequence_3d(batch_axis, time_axis):
    np.random.seed(0)
    x = np.random.randn(4, 4, 3).astype(np.float32)
    sequence_lens = np.array([4, 3, 2, 1], dtype=np.int64)
    check_reverse_sequence(x, sequence_lens, batch_axis=batch_axis, time_axis=time_axis)
