"""GRU, LSTM and RNN attributes that used to be rejected at conversion."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from torch import nn

from onnx2pytorch.convert import ConvertModel
from onnx2pytorch.operations.gru import GRUWrapper
from onnx2pytorch.operations.lstm import LSTMWrapper
from onnx2pytorch.operations.recurrent import OnnxRecurrent
from onnx2pytorch.operations.rnn import RNNWrapper

RTOL = 1e-4
ATOL = 1e-5


def _model(nodes, inputs, outputs, initializers, opset=14):
    graph = helper.make_graph(nodes, "g", inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    return model


def _vi(name, shape, dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, shape)


def _check(model, feeds):
    payload = model.SerializeToString()
    expected = ort.InferenceSession(payload, providers=["CPUExecutionProvider"]).run(
        None, feeds
    )
    converted = ConvertModel(onnx.load_from_string(payload))
    converted.eval()
    names = [value.name for value in onnx.load_from_string(payload).graph.input]
    with torch.no_grad():
        actual = converted(
            *[torch.from_numpy(np.asarray(feeds[name])) for name in names]
        )
    if not isinstance(actual, (list, tuple)):
        actual = (actual,)
    diffs = []
    for got, want in zip(actual, expected):
        got = got.detach().cpu().numpy()
        np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)
        diffs.append(
            float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64))))
        )
    return converted, diffs


def _recurrent(
    op_type,
    *,
    direction="forward",
    layout=0,
    activations=None,
    alpha=None,
    beta=None,
    clip=None,
    linear_before_reset=None,
    input_forget=None,
    with_seq=False,
    with_state=False,
    with_peephole=False,
    seed=0
):
    rng = np.random.default_rng(seed)
    gates = {"RNN": 1, "GRU": 3, "LSTM": 4}[op_type]
    seq, batch, input_size, hidden = 5, 2, 4, 3
    directions = 2 if direction == "bidirectional" else 1
    weights = [
        numpy_helper.from_array(
            (
                rng.standard_normal((directions, gates * hidden, input_size)) * 0.5
            ).astype(np.float32),
            "W",
        ),
        numpy_helper.from_array(
            (rng.standard_normal((directions, gates * hidden, hidden)) * 0.5).astype(
                np.float32
            ),
            "R",
        ),
        numpy_helper.from_array(
            (rng.standard_normal((directions, 2 * gates * hidden)) * 0.5).astype(
                np.float32
            ),
            "B",
        ),
    ]
    inputs = ["X", "W", "R", "B"]
    graph_inputs = [
        _vi("X", [batch, seq, input_size] if layout == 1 else [seq, batch, input_size])
    ]
    feeds = {
        "X": rng.standard_normal(
            graph_inputs[0].type.tensor_type.shape.dim and ()
        ).astype(np.float32)
    }
    # rebuild X from the concrete shape
    x_shape = [batch, seq, input_size] if layout == 1 else [seq, batch, input_size]
    feeds["X"] = rng.standard_normal(x_shape).astype(np.float32)
    graph_inputs = [_vi("X", x_shape)]

    if with_seq:
        inputs += ["seq"]
        lengths = np.array([seq, 0], dtype=np.int32)
        weights.append(numpy_helper.from_array(lengths, "seq"))
    else:
        inputs += [""]

    state_shape = (
        [batch, directions, hidden] if layout == 1 else [directions, batch, hidden]
    )
    if with_state:
        inputs.append("H")
        graph_inputs.append(_vi("H", state_shape))
        feeds["H"] = rng.standard_normal(state_shape).astype(np.float32)
        if op_type == "LSTM":
            inputs.append("C")
            graph_inputs.append(_vi("C", state_shape))
            feeds["C"] = rng.standard_normal(state_shape).astype(np.float32)
    if with_peephole:
        while len(inputs) < 7:
            inputs.append("")
        inputs.append("P")
        weights.append(
            numpy_helper.from_array(
                (rng.standard_normal((directions, 3 * hidden)) * 0.5).astype(
                    np.float32
                ),
                "P",
            )
        )

    outputs = ["Y", "Y_h"] + (["Y_c"] if op_type == "LSTM" else [])
    attrs = {"hidden_size": hidden, "direction": direction, "layout": layout}
    if activations is not None:
        attrs["activations"] = activations
    if alpha is not None:
        attrs["activation_alpha"] = alpha
    if beta is not None:
        attrs["activation_beta"] = beta
    if clip is not None:
        attrs["clip"] = clip
    if linear_before_reset is not None:
        attrs["linear_before_reset"] = linear_before_reset
    if input_forget is not None:
        attrs["input_forget"] = input_forget
    node = helper.make_node(op_type, inputs, outputs, **attrs)
    model = _model(
        [node],
        graph_inputs,
        [_vi(name, None) for name in outputs],
        weights,
    )
    return model, feeds


def _kind(converted):
    for module in converted.modules():
        if isinstance(module, (GRUWrapper, LSTMWrapper, RNNWrapper, OnnxRecurrent)):
            return module
    raise AssertionError("no recurrent module")


def _torch_cell(module, op_type):
    if op_type == "GRU":
        return module.gru
    if op_type == "LSTM":
        return module.lstm
    return module.rnn


@pytest.mark.parametrize("op_type", ["GRU", "LSTM", "RNN"])
def test_default_attributes_stay_on_torch_modules(op_type):
    model, feeds = _recurrent(op_type)
    converted, _ = _check(model, feeds)
    module = _kind(converted)
    assert not isinstance(module, OnnxRecurrent)
    inner = _torch_cell(module, op_type)
    assert isinstance(inner, {"GRU": nn.GRU, "LSTM": nn.LSTM, "RNN": nn.RNN}[op_type])


@pytest.mark.parametrize(
    "op_type, activations",
    [
        ("GRU", ["Sigmoid", "Tanh"]),
        ("GRU", ["sigmoid", "tanh"]),
        ("LSTM", ["Sigmoid", "Tanh", "Tanh"]),
        ("RNN", ["Tanh"]),
        ("RNN", ["Relu"]),
    ],
)
def test_explicit_default_activations_use_torch(op_type, activations):
    model, feeds = _recurrent(op_type, activations=activations)
    converted, _ = _check(model, feeds)
    module = _kind(converted)
    assert not isinstance(module, OnnxRecurrent)
    if op_type == "RNN" and activations == ["Relu"]:
        assert module.rnn.nonlinearity == "relu"


@pytest.mark.parametrize("op_type", ["GRU", "LSTM", "RNN"])
def test_bidirectional_repeated_default_activations(op_type):
    defaults = {
        "GRU": ["Sigmoid", "Tanh"],
        "LSTM": ["Sigmoid", "Tanh", "Tanh"],
        "RNN": ["Tanh"],
    }
    model, feeds = _recurrent(
        op_type, direction="bidirectional", activations=defaults[op_type] * 2
    )
    converted, _ = _check(model, feeds)
    assert not isinstance(_kind(converted), OnnxRecurrent)


def _layout0_reference(op_type, model, feeds):
    """onnxruntime 1.30's CPU kernels reject layout=1.

    The same weights with layout 0, after the spec's batch/seq transposes, are
    the reference for the batch-first outputs.
    """
    graph = model.graph
    node = onnx.NodeProto()
    node.CopyFrom(graph.node[0])
    for attr in node.attribute:
        if attr.name == "layout":
            attr.i = 0
    seq_first = helper.make_node("Transpose", ["X"], ["X_seq"], perm=[1, 0, 2])
    node.input[0] = "X_seq"
    state_nodes = []
    for name in ("H", "C"):
        if name not in feeds:
            continue
        seq_name = name + "_seq"
        state_nodes.append(
            helper.make_node("Transpose", [name], [seq_name], perm=[1, 0, 2])
        )
        for index, input_name in enumerate(node.input):
            if input_name == name:
                node.input[index] = seq_name
    ref_outputs = []
    back = []
    node.output[:] = [name + "_seq" for name in node.output]
    for output in graph.output:
        seq_name = output.name + "_seq"
        ref_outputs.append(_vi(seq_name, None))
        # Y: (seq, dir, batch, hidden) -> (batch, seq, dir, hidden)
        # Y_h / Y_c: (dir, batch, hidden) -> (batch, dir, hidden)
        perm = [2, 0, 1, 3] if output.name == "Y" else [1, 0, 2]
        back.append(helper.make_node("Transpose", [seq_name], [output.name], perm=perm))
    ref = _model(
        [seq_first, *state_nodes, node, *back],
        list(graph.input),
        list(graph.output),
        list(graph.initializer),
    )
    return ort.InferenceSession(
        ref.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, feeds)


@pytest.mark.parametrize("op_type", ["GRU", "LSTM", "RNN"])
def test_layout_batch_first(op_type):
    model, feeds = _recurrent(op_type, layout=1, with_state=True)
    expected = _layout0_reference(op_type, model, feeds)
    converted = ConvertModel(model)
    converted.eval()
    names = [value.name for value in model.graph.input]
    with torch.no_grad():
        actual = converted(*[torch.from_numpy(feeds[name]) for name in names])
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(
            got.detach().cpu().numpy(), want, rtol=RTOL, atol=ATOL
        )


@pytest.mark.parametrize("op_type", ["GRU", "LSTM", "RNN"])
@pytest.mark.parametrize("direction", ["forward", "reverse", "bidirectional"])
def test_clip(op_type, direction):
    model, feeds = _recurrent(op_type, direction=direction, clip=0.3, with_state=True)
    converted, _ = _check(model, feeds)
    assert isinstance(_kind(converted), OnnxRecurrent)


def test_gru_nontorch_activations_with_alpha_beta():
    model, feeds = _recurrent(
        "GRU",
        activations=["HardSigmoid", "Softsign"],
        alpha=[0.3],
        beta=[0.4],
        linear_before_reset=0,
    )
    converted, _ = _check(model, feeds)
    assert isinstance(_kind(converted), OnnxRecurrent)
    model, feeds = _recurrent(
        "GRU",
        direction="bidirectional",
        activations=["HardSigmoid", "Softsign", "Sigmoid", "Tanh"],
        alpha=[0.25, 0.0, 1.0, 1.0],
        beta=[0.6, 0.0, 0.0, 0.0],
        linear_before_reset=1,
        seed=1,
    )
    _check(model, feeds)


@pytest.mark.parametrize("linear_before_reset", [0, 1])
def test_gru_linear_before_reset(linear_before_reset):
    model, feeds = _recurrent(
        "GRU",
        direction="bidirectional",
        linear_before_reset=linear_before_reset,
        with_state=True,
        with_seq=True,
    )
    _check(model, feeds)


def test_lstm_initial_state_and_sequence_lens():
    model, feeds = _recurrent(
        "LSTM", direction="bidirectional", with_state=True, with_seq=True, layout=0
    )
    _check(model, feeds)


def test_lstm_peephole_and_input_forget():
    model, feeds = _recurrent(
        "LSTM", with_peephole=True, with_state=True, clip=0.4, seed=2
    )
    _check(model, feeds)
    model, feeds = _recurrent("LSTM", input_forget=1, with_state=True, seed=3)
    _check(model, feeds)


def test_unknown_activation_names_the_attribute():
    model, _ = _recurrent("LSTM", activations=["NotAnActivation", "Tanh", "Tanh"])
    with pytest.raises(
        NotImplementedError, match=r"LSTM activations \[.NotAnActivation."
    ):
        ConvertModel(model)


def test_unsupported_layout_names_the_attribute():
    model, _ = _recurrent("GRU", layout=2)
    with pytest.raises(NotImplementedError, match=r"GRU layout=2"):
        ConvertModel(model)
