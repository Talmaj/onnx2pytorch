import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_sequence_insert(tensors, new_tensor, position=None, start_empty=False):
    names = ["t%d" % i for i in range(len(tensors))]
    if start_empty:
        nodes = [
            helper.make_node("SequenceEmpty", [], ["seq"], dtype=TensorProto.FLOAT)
        ]
    else:
        nodes = [helper.make_node("SequenceConstruct", names, ["seq"])]
    insert_inputs = ["seq", "new"]
    if position is not None:
        insert_inputs.append("position")
    nodes.append(helper.make_node("SequenceInsert", insert_inputs, ["seq_out"]))

    graph_inputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, list(t.shape))
        for name, t in zip(names, tensors)
    ]
    feeds = dict(zip(names, tensors))
    graph_inputs.append(
        helper.make_tensor_value_info("new", TensorProto.FLOAT, list(new_tensor.shape))
    )
    feeds["new"] = new_tensor
    if position is not None:
        graph_inputs.append(
            helper.make_tensor_value_info(
                "position", TensorProto.INT64, list(position.shape)
            )
        )
        feeds["position"] = position

    graph = helper.make_graph(
        nodes,
        "sequenceinsert_test",
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
    return [np.full((2, 3), i, dtype=np.float32) for i in range(count)]


def make_new_tensor():
    return np.full((2, 3), 9.0, dtype=np.float32)


def test_sequence_insert_default_position():
    check_sequence_insert(make_tensors(), make_new_tensor())


@pytest.mark.parametrize("position", [0, 1, 2, 3])
def test_sequence_insert_positive_position(position):
    check_sequence_insert(
        make_tensors(), make_new_tensor(), np.array(position, dtype=np.int64)
    )


@pytest.mark.parametrize("position", [-1, -2, -3])
def test_sequence_insert_negative_position(position):
    check_sequence_insert(
        make_tensors(), make_new_tensor(), np.array(position, dtype=np.int64)
    )


def test_sequence_insert_into_empty_sequence():
    check_sequence_insert(make_tensors(count=0), make_new_tensor(), start_empty=True)


def test_sequence_insert_position_as_1d_tensor():
    check_sequence_insert(
        make_tensors(), make_new_tensor(), np.array([1], dtype=np.int64)
    )


def test_sequence_insert_single_element_sequence():
    check_sequence_insert(
        make_tensors(count=1), make_new_tensor(), np.array(0, dtype=np.int64)
    )
