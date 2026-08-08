import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_sequence_map(nodes, body_outputs, feeds, shapes):
    graph = helper.make_graph(
        nodes,
        "sequencemap_test",
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, shapes[name])
            for name in feeds
        ],
        [
            helper.make_tensor_sequence_value_info(name, TensorProto.FLOAT, None)
            for name in body_outputs
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    expected = ort.InferenceSession(model.SerializeToString()).run(None, feeds)

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        res = o2p_model(**{k: torch.from_numpy(v) for k, v in feeds.items()})
    if len(body_outputs) == 1:
        res = [res]

    for actual_seq, exp_seq in zip(res, expected):
        assert len(actual_seq) == len(exp_seq)
        for actual, exp in zip(actual_seq, exp_seq):
            np.testing.assert_allclose(actual.numpy(), exp, rtol=1e-5, atol=1e-5)


def make_body(nodes, input_names, output_names):
    return helper.make_graph(
        nodes,
        "body",
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
            for name in input_names
        ],
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
            for name in output_names
        ],
    )


def test_sequence_map_identity():
    body = make_body(
        [helper.make_node("Identity", ["in0"], ["out0"])], ["in0"], ["out0"]
    )
    nodes = [
        helper.make_node("SplitToSequence", ["x"], ["seq"], axis=0),
        helper.make_node("SequenceMap", ["seq"], ["seq_out"], body=body),
    ]
    np.random.seed(0)
    x = np.random.randn(3, 4).astype(np.float32)
    check_sequence_map(nodes, ["seq_out"], {"x": x}, {"x": [3, 4]})


def test_sequence_map_double():
    body = make_body(
        [helper.make_node("Add", ["in0", "in0"], ["out0"])], ["in0"], ["out0"]
    )
    nodes = [
        helper.make_node("SplitToSequence", ["x"], ["seq"], axis=0, keepdims=0),
        helper.make_node("SequenceMap", ["seq"], ["seq_out"], body=body),
    ]
    np.random.seed(1)
    x = np.random.randn(4, 2).astype(np.float32)
    check_sequence_map(nodes, ["seq_out"], {"x": x}, {"x": [4, 2]})


def test_sequence_map_from_sequence_construct():
    body = make_body(
        [helper.make_node("Mul", ["in0", "in0"], ["out0"])], ["in0"], ["out0"]
    )
    nodes = [
        helper.make_node("SequenceConstruct", ["a", "b", "c"], ["seq"]),
        helper.make_node("SequenceMap", ["seq"], ["seq_out"], body=body),
    ]
    np.random.seed(2)
    feeds = {
        "a": np.random.randn(2, 3).astype(np.float32),
        "b": np.random.randn(4, 5).astype(np.float32),
        "c": np.random.randn(1).astype(np.float32),
    }
    shapes = {"a": [2, 3], "b": [4, 5], "c": [1]}
    check_sequence_map(nodes, ["seq_out"], feeds, shapes)


def test_sequence_map_additional_tensor_input():
    body = make_body(
        [helper.make_node("Add", ["in0", "extra"], ["out0"])],
        ["in0", "extra"],
        ["out0"],
    )
    nodes = [
        helper.make_node("SplitToSequence", ["x"], ["seq"], axis=0, keepdims=0),
        helper.make_node("SequenceMap", ["seq", "extra"], ["seq_out"], body=body),
    ]
    np.random.seed(3)
    feeds = {
        "x": np.random.randn(3, 4).astype(np.float32),
        "extra": np.random.randn(4).astype(np.float32),
    }
    check_sequence_map(nodes, ["seq_out"], feeds, {"x": [3, 4], "extra": [4]})


def test_sequence_map_additional_sequence_input():
    body = make_body(
        [helper.make_node("Add", ["in0", "in1"], ["out0"])], ["in0", "in1"], ["out0"]
    )
    nodes = [
        helper.make_node("SplitToSequence", ["x"], ["seq_x"], axis=0, keepdims=0),
        helper.make_node("SplitToSequence", ["y"], ["seq_y"], axis=0, keepdims=0),
        helper.make_node("SequenceMap", ["seq_x", "seq_y"], ["seq_out"], body=body),
    ]
    np.random.seed(4)
    feeds = {
        "x": np.random.randn(3, 4).astype(np.float32),
        "y": np.random.randn(3, 4).astype(np.float32),
    }
    check_sequence_map(nodes, ["seq_out"], feeds, {"x": [3, 4], "y": [3, 4]})


def test_sequence_map_multiple_outputs():
    body = make_body(
        [
            helper.make_node("Add", ["in0", "in0"], ["out0"]),
            helper.make_node("Mul", ["in0", "in0"], ["out1"]),
        ],
        ["in0"],
        ["out0", "out1"],
    )
    nodes = [
        helper.make_node("SplitToSequence", ["x"], ["seq"], axis=0, keepdims=0),
        helper.make_node("SequenceMap", ["seq"], ["seq_a", "seq_b"], body=body),
    ]
    np.random.seed(5)
    x = np.random.randn(3, 2).astype(np.float32)
    check_sequence_map(nodes, ["seq_a", "seq_b"], {"x": x}, {"x": [3, 2]})


def test_sequence_map_multi_node_body():
    body = make_body(
        [
            helper.make_node("Add", ["in0", "in0"], ["doubled"]),
            helper.make_node("Relu", ["doubled"], ["out0"]),
        ],
        ["in0"],
        ["out0"],
    )
    nodes = [
        helper.make_node("SplitToSequence", ["x"], ["seq"], axis=1, keepdims=0),
        helper.make_node("SequenceMap", ["seq"], ["seq_out"], body=body),
    ]
    np.random.seed(6)
    x = np.random.randn(2, 5).astype(np.float32)
    check_sequence_map(nodes, ["seq_out"], {"x": x}, {"x": [2, 5]})


def test_sequence_map_body_with_initializer():
    from onnx import numpy_helper

    offset = np.array([1.0, 2.0], dtype=np.float32)
    body = helper.make_graph(
        [helper.make_node("Add", ["in0", "offset"], ["out0"])],
        "body",
        [helper.make_tensor_value_info("in0", TensorProto.FLOAT, None)],
        [helper.make_tensor_value_info("out0", TensorProto.FLOAT, None)],
        initializer=[numpy_helper.from_array(offset, "offset")],
    )
    nodes = [
        helper.make_node("SplitToSequence", ["x"], ["seq"], axis=0, keepdims=0),
        helper.make_node("SequenceMap", ["seq"], ["seq_out"], body=body),
    ]
    np.random.seed(7)
    x = np.random.randn(3, 2).astype(np.float32)
    check_sequence_map(nodes, ["seq_out"], {"x": x}, {"x": [3, 2]})
