import io

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import helper, numpy_helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def build_rnn_model(input_size, hidden_size, direction, with_bias, activations=None):
    num_directions = 2 if direction == "bidirectional" else 1
    rng = np.random.RandomState(0)
    w = rng.randn(num_directions, hidden_size, input_size).astype(np.float32)
    r = rng.randn(num_directions, hidden_size, hidden_size).astype(np.float32)
    initializers = [numpy_helper.from_array(w, "W"), numpy_helper.from_array(r, "R")]
    inputs = ["X", "W", "R"]
    if with_bias:
        b = rng.randn(num_directions, 2 * hidden_size).astype(np.float32)
        initializers.append(numpy_helper.from_array(b, "B"))
        inputs.append("B")

    attrs = {"hidden_size": hidden_size, "direction": direction}
    if activations is not None:
        attrs["activations"] = activations

    node = helper.make_node("RNN", inputs=inputs, outputs=["Y", "Y_h"], **attrs)
    graph = helper.make_graph(
        [node],
        "rnn_test",
        [
            helper.make_tensor_value_info(
                "X", TensorProto.FLOAT, [None, None, input_size]
            )
        ],
        [
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, None),
        ],
        initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])


def check_rnn(model, x):
    exp = ort.InferenceSession(model.SerializeToString()).run(None, {"X": x})
    with torch.no_grad():
        y, y_h = ConvertModel(model)(torch.from_numpy(x))
    np.testing.assert_allclose(y.numpy(), exp[0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(y_h.numpy(), exp[1], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("direction", ["forward", "bidirectional"])
@pytest.mark.parametrize("with_bias", [True, False])
def test_rnn(direction, with_bias):
    np.random.seed(1)
    x = np.random.randn(7, 4, 3).astype(np.float32)
    check_rnn(build_rnn_model(3, 5, direction, with_bias), x)


@pytest.mark.parametrize("direction", ["forward", "bidirectional"])
def test_rnn_relu_activation(direction):
    np.random.seed(1)
    x = np.random.randn(6, 2, 3).astype(np.float32)
    activations = ["Relu"] * (2 if direction == "bidirectional" else 1)
    check_rnn(build_rnn_model(3, 4, direction, True, activations), x)


def test_rnn_unsupported_activation():
    model = build_rnn_model(3, 4, "forward", True, ["Sigmoid"])
    with pytest.raises(NotImplementedError):
        ConvertModel(model)


def test_rnn_exported_from_torch():
    torch.manual_seed(42)
    rnn = torch.nn.RNN(input_size=3, hidden_size=5, num_layers=1)
    input = torch.randn(11, 4, 3)
    h_0 = torch.randn(1, 4, 5)
    output, h_n = rnn(input, h_0)

    bitstream = io.BytesIO()
    torch.onnx.export(
        model=rnn,
        args=(input, h_0),
        f=bitstream,
        input_names=["input", "h_0"],
        opset_version=11,
        dynamo=False,
    )
    onnx_rnn = onnx.ModelProto.FromString(bitstream.getvalue())

    o2p_rnn = ConvertModel(onnx_rnn)
    with torch.no_grad():
        o2p_output, o2p_h_n = o2p_rnn(input, h_0)
    torch.testing.assert_close(o2p_output, output, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(o2p_h_n, h_n, rtol=1e-6, atol=1e-6)
