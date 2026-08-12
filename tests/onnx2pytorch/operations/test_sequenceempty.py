import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel
from onnx2pytorch.operations import SequenceEmpty


def check_sequence_empty(x, **attrs):
    """The empty sequence is a graph output next to a plain tensor output."""
    nodes = [
        helper.make_node("SequenceEmpty", [], ["seq"], **attrs),
        helper.make_node("Identity", ["x"], ["y"]),
    ]
    elem_type = attrs.get("dtype", TensorProto.FLOAT)
    graph = helper.make_graph(
        nodes,
        "sequenceempty_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [
            helper.make_tensor_sequence_value_info("seq", elem_type, None),
            helper.make_tensor_value_info("y", TensorProto.FLOAT, None),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    exp_seq, exp_y = ort.InferenceSession(model.SerializeToString()).run(None, {"x": x})

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        seq, y = o2p_model(torch.from_numpy(x))

    assert isinstance(seq, list)
    assert len(seq) == len(exp_seq) == 0
    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-5, atol=1e-5)


def test_sequence_empty_default_dtype():
    check_sequence_empty(np.array([1.0, 2.0], dtype=np.float32))


@pytest.mark.parametrize(
    "dtype", [TensorProto.FLOAT, TensorProto.INT64, TensorProto.DOUBLE]
)
def test_sequence_empty_dtype(dtype):
    check_sequence_empty(np.array([1.0, 2.0], dtype=np.float32), dtype=dtype)


def test_sequence_empty_module_returns_empty_list():
    assert SequenceEmpty()() == []
    assert SequenceEmpty(dtype=TensorProto.INT64)() == []
