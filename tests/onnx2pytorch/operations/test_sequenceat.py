import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_sequence_at(tensors, position):
    names = ["t%d" % i for i in range(len(tensors))]
    nodes = [
        helper.make_node("SequenceConstruct", names, ["seq"]),
        helper.make_node("SequenceAt", ["seq", "position"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "sequenceat_test",
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, list(t.shape))
            for name, t in zip(names, tensors)
        ]
        + [
            helper.make_tensor_value_info(
                "position", TensorProto.INT64, list(position.shape)
            )
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    feeds = dict(zip(names, tensors))
    feeds["position"] = position
    exp_y = ort.InferenceSession(model.SerializeToString()).run(None, feeds)[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(**{k: torch.from_numpy(v) for k, v in feeds.items()})

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-5, atol=1e-5)


def make_tensors(count=3):
    np.random.seed(0)
    return [np.random.randn(2, 3).astype(np.float32) for _ in range(count)]


@pytest.mark.parametrize("position", [0, 1, 2])
def test_sequence_at_positive_position(position):
    check_sequence_at(make_tensors(), np.array(position, dtype=np.int64))


@pytest.mark.parametrize("position", [-1, -2, -3])
def test_sequence_at_negative_position(position):
    check_sequence_at(make_tensors(), np.array(position, dtype=np.int64))


def test_sequence_at_single_element():
    check_sequence_at(make_tensors(count=1), np.array(0, dtype=np.int64))


def test_sequence_at_position_as_1d_tensor():
    check_sequence_at(make_tensors(), np.array([1], dtype=np.int64))


def test_sequence_at_different_shapes():
    np.random.seed(1)
    tensors = [
        np.random.randn(2, 3).astype(np.float32),
        np.random.randn(4, 5, 6).astype(np.float32),
    ]
    check_sequence_at(tensors, np.array(1, dtype=np.int64))
