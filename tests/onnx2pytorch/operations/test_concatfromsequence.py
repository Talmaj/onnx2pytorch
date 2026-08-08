import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_concat_from_sequence(tensors, **attrs):
    names = ["t%d" % i for i in range(len(tensors))]
    nodes = [
        helper.make_node("SequenceConstruct", names, ["seq"]),
        helper.make_node("ConcatFromSequence", ["seq"], ["y"], **attrs),
    ]
    graph = helper.make_graph(
        nodes,
        "concatfromsequence_test",
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, list(t.shape))
            for name, t in zip(names, tensors)
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    feeds = dict(zip(names, tensors))
    exp_y = ort.InferenceSession(model.SerializeToString()).run(None, feeds)[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(**{k: torch.from_numpy(v) for k, v in feeds.items()})

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("axis", [0, 1, -1, -2])
def test_concat_from_sequence_concat(axis):
    np.random.seed(0)
    tensors = [np.random.randn(2, 3).astype(np.float32) for _ in range(3)]
    check_concat_from_sequence(tensors, axis=axis)


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_concat_from_sequence_new_axis(axis):
    np.random.seed(1)
    tensors = [np.random.randn(2, 3).astype(np.float32) for _ in range(3)]
    check_concat_from_sequence(tensors, axis=axis, new_axis=1)


def test_concat_from_sequence_different_lengths():
    np.random.seed(2)
    tensors = [
        np.random.randn(1, 3).astype(np.float32),
        np.random.randn(4, 3).astype(np.float32),
        np.random.randn(2, 3).astype(np.float32),
    ]
    check_concat_from_sequence(tensors, axis=0)


def test_concat_from_sequence_single_element():
    np.random.seed(3)
    check_concat_from_sequence([np.random.randn(2, 3).astype(np.float32)], axis=0)


def test_concat_from_sequence_3d():
    np.random.seed(4)
    tensors = [np.random.randn(2, 3, 4).astype(np.float32) for _ in range(2)]
    check_concat_from_sequence(tensors, axis=1)
