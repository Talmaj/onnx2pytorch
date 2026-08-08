import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

from onnx2pytorch.convert import ConvertModel


def check_split_to_sequence(x, split=None, use_reference=False, **attrs):
    inputs = ["x"] if split is None else ["x", "split"]
    node = helper.make_node("SplitToSequence", inputs, ["seq"], **attrs)
    graph_inputs = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))
    ]
    feeds = {"x": x}
    if split is not None:
        graph_inputs.append(
            helper.make_tensor_value_info("split", TensorProto.INT64, list(split.shape))
        )
        feeds["split"] = split

    graph = helper.make_graph(
        [node],
        "splittosequence_test",
        graph_inputs,
        [helper.make_tensor_sequence_value_info("seq", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    if use_reference:
        exp_seq = ReferenceEvaluator(model).run(None, feeds)[0]
    else:
        exp_seq = ort.InferenceSession(model.SerializeToString()).run(None, feeds)[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        seq = o2p_model(**{k: torch.from_numpy(v) for k, v in feeds.items()})

    assert len(seq) == len(exp_seq)
    for actual, exp in zip(seq, exp_seq):
        assert actual.shape == exp.shape
        np.testing.assert_allclose(actual.numpy(), exp, rtol=1e-5, atol=1e-5)


def make_input(shape=(4, 3), seed=0):
    np.random.seed(seed)
    return np.random.randn(*shape).astype(np.float32)


@pytest.mark.parametrize("axis", [0, 1, -1, -2])
def test_split_to_sequence_default_split(axis):
    check_split_to_sequence(make_input(), axis=axis)


@pytest.mark.parametrize("keepdims", [0, 1])
def test_split_to_sequence_keepdims(keepdims):
    check_split_to_sequence(make_input(seed=1), axis=0, keepdims=keepdims)


def test_split_to_sequence_keepdims_on_second_axis():
    check_split_to_sequence(make_input(seed=2), axis=1, keepdims=0)


@pytest.mark.parametrize("chunk", [1, 2, 3, 4])
def test_split_to_sequence_scalar_split(chunk):
    check_split_to_sequence(
        make_input(shape=(5, 3), seed=3), split=np.array(chunk, dtype=np.int64)
    )


def test_split_to_sequence_split_lengths():
    check_split_to_sequence(
        make_input(shape=(6, 3), seed=4), split=np.array([1, 2, 3], dtype=np.int64)
    )


def test_split_to_sequence_split_lengths_on_axis_one():
    check_split_to_sequence(
        make_input(shape=(2, 6), seed=5),
        split=np.array([4, 2], dtype=np.int64),
        axis=1,
    )


def test_split_to_sequence_keepdims_ignored_with_split():
    """keepdims is ignored when split is given; onnxruntime squeezes anyway."""
    check_split_to_sequence(
        make_input(shape=(4, 3), seed=6),
        split=np.array(1, dtype=np.int64),
        keepdims=0,
        use_reference=True,
    )


def test_split_to_sequence_3d():
    check_split_to_sequence(make_input(shape=(2, 3, 4), seed=7), axis=2, keepdims=0)
