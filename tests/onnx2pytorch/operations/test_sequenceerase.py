import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_sequence_erase(tensors, position=None):
    names = ["t%d" % i for i in range(len(tensors))]
    erase_inputs = ["seq"] if position is None else ["seq", "position"]
    nodes = [
        helper.make_node("SequenceConstruct", names, ["seq"]),
        helper.make_node("SequenceErase", erase_inputs, ["seq_out"]),
    ]
    graph_inputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, list(t.shape))
        for name, t in zip(names, tensors)
    ]
    feeds = dict(zip(names, tensors))
    if position is not None:
        graph_inputs.append(
            helper.make_tensor_value_info(
                "position", TensorProto.INT64, list(position.shape)
            )
        )
        feeds["position"] = position

    graph = helper.make_graph(
        nodes,
        "sequenceerase_test",
        graph_inputs,
        [helper.make_tensor_sequence_value_info("seq_out", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    exp_seq = ort.InferenceSession(model.SerializeToString()).run(None, feeds)[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        seq = o2p_model(**{k: torch.from_numpy(v) for k, v in feeds.items()})

    assert len(seq) == len(exp_seq)
    for actual, exp in zip(seq, exp_seq):
        np.testing.assert_allclose(actual.numpy(), exp, rtol=1e-5, atol=1e-5)


def make_tensors(count=3):
    np.random.seed(0)
    return [np.full((2, 3), i, dtype=np.float32) for i in range(count)]


def test_sequence_erase_default_position():
    check_sequence_erase(make_tensors())


@pytest.mark.parametrize("position", [0, 1, 2])
def test_sequence_erase_positive_position(position):
    check_sequence_erase(make_tensors(), np.array(position, dtype=np.int64))


@pytest.mark.parametrize("position", [-1, -2, -3])
def test_sequence_erase_negative_position(position):
    check_sequence_erase(make_tensors(), np.array(position, dtype=np.int64))


def test_sequence_erase_to_empty():
    check_sequence_erase(make_tensors(count=1), np.array(0, dtype=np.int64))


def test_sequence_erase_position_as_1d_tensor():
    check_sequence_erase(make_tensors(), np.array([1], dtype=np.int64))
