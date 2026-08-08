import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_sequence_length(tensors, start_empty=False):
    names = ["t%d" % i for i in range(len(tensors))]
    if start_empty:
        nodes = [
            helper.make_node("SequenceEmpty", [], ["seq"], dtype=TensorProto.FLOAT),
            helper.make_node("Identity", ["t0"], ["ignored"]),
        ]
    else:
        nodes = [helper.make_node("SequenceConstruct", names, ["seq"])]
    nodes.append(helper.make_node("SequenceLength", ["seq"], ["length"]))

    graph = helper.make_graph(
        nodes,
        "sequencelength_test",
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, list(t.shape))
            for name, t in zip(names, tensors)
        ],
        [helper.make_tensor_value_info("length", TensorProto.INT64, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    feeds = dict(zip(names, tensors))
    exp_length = ort.InferenceSession(model.SerializeToString()).run(None, feeds)[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        length = o2p_model(**{k: torch.from_numpy(v) for k, v in feeds.items()})

    assert length.dtype == torch.int64
    np.testing.assert_array_equal(length.numpy(), exp_length)


@pytest.mark.parametrize("count", [1, 2, 5])
def test_sequence_length(count):
    np.random.seed(0)
    tensors = [np.random.randn(2, 3).astype(np.float32) for _ in range(count)]
    check_sequence_length(tensors)


def test_sequence_length_empty_sequence():
    tensors = [np.zeros((2, 3), dtype=np.float32)]
    check_sequence_length(tensors, start_empty=True)


def test_sequence_length_different_shapes():
    np.random.seed(1)
    tensors = [
        np.random.randn(2).astype(np.float32),
        np.random.randn(3, 4).astype(np.float32),
        np.random.randn(1, 1, 1).astype(np.float32),
    ]
    check_sequence_length(tensors)
