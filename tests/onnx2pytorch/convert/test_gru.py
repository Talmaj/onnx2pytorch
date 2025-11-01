import io
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


@pytest.mark.parametrize(
    "bidirectional, input_size, hidden_size, seq_len, batch, test_seq_len, test_batch",
    [
        (False, 3, 5, 23, 4, 23, 4),
        (False, 3, 5, 23, 4, 37, 4),
        (False, 3, 5, 23, 4, 23, 7),
        (True, 3, 5, 23, 4, 23, 4),
        (True, 3, 5, 23, 4, 37, 4),
        (True, 3, 5, 23, 4, 23, 7),
    ],
)
def test_single_layer_gru(
    bidirectional, input_size, hidden_size, seq_len, batch, test_seq_len, test_batch
):
    torch.manual_seed(42)
    num_layers = 1
    num_directions = bidirectional + 1
    gru = torch.nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )
    input = torch.randn(seq_len, batch, input_size)
    h_0 = torch.randn(num_layers * num_directions, batch, hidden_size)
    output, h_n = gru(input, h_0)
    bitstream = io.BytesIO()
    torch.onnx.export(
        model=gru,
        args=(input, h_0),
        f=bitstream,
        input_names=["input", "h_0"],
        opset_version=11,
        dynamo=False,  # Use legacy exporter for GRU compatibility
        dynamic_axes={
            "input": {0: "seq_len", 1: "batch"},
            "h_0": {1: "batch"},
        },
    )
    bitstream_data = bitstream.getvalue()

    onnx_gru = onnx.ModelProto.FromString(bitstream_data)
    o2p_gru = ConvertModel(onnx_gru, experimental=True)
    with torch.no_grad():
        o2p_output, o2p_h_n = o2p_gru(input, h_0)
        torch.testing.assert_close(o2p_output, output, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(o2p_h_n, h_n, rtol=1e-6, atol=1e-6)

    onnx_gru = onnx.ModelProto.FromString(bitstream_data)
    o2p_gru = ConvertModel(onnx_gru, experimental=True)
    with torch.no_grad():
        o2p_output, o2p_h_n = o2p_gru(h_0=h_0, input=input)
        torch.testing.assert_close(o2p_output, output, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(o2p_h_n, h_n, rtol=1e-6, atol=1e-6)
        with pytest.raises(KeyError):
            o2p_output, o2p_h_n = o2p_gru(input=input)
        with pytest.raises(Exception):
            # Even though initial states are optional for nn.GRU(),
            # we adhere to onnxruntime convention that inputs are provided
            # as either all positional or all keyword arguments.
            o2p_output, o2p_h_n = o2p_gru(input, h_0=h_0)


@pytest.mark.parametrize("linear_before_reset", [0, 1])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_gru_linear_before_reset(linear_before_reset, bidirectional):
    """Test GRU with both linear_before_reset=0 (ONNX/TensorFlow default) and =1 (PyTorch default)."""
    torch.manual_seed(42)
    np.random.seed(42)

    input_size = 3
    hidden_size = 4
    seq_len = 5
    batch = 2
    num_directions = 2 if bidirectional else 1

    # Create input and initial hidden state
    X = np.random.randn(seq_len, batch, input_size).astype(np.float32)
    initial_h = np.random.randn(num_directions, batch, hidden_size).astype(np.float32)

    # Create random weights for GRU
    # W shape: [num_directions, 3*hidden_size, input_size]
    W = np.random.randn(num_directions, 3 * hidden_size, input_size).astype(np.float32)
    # R shape: [num_directions, 3*hidden_size, hidden_size]
    R = np.random.randn(num_directions, 3 * hidden_size, hidden_size).astype(np.float32)
    # B shape: [num_directions, 6*hidden_size] (Wb and Rb concatenated)
    B = np.random.randn(num_directions, 6 * hidden_size).astype(np.float32)

    # Create ONNX graph with GRU node
    input_tensor = helper.make_tensor_value_info(
        "X", TensorProto.FLOAT, [seq_len, batch, input_size]
    )
    initial_h_tensor = helper.make_tensor_value_info(
        "initial_h", TensorProto.FLOAT, [num_directions, batch, hidden_size]
    )
    output_tensor = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, [seq_len, num_directions, batch, hidden_size]
    )
    output_h_tensor = helper.make_tensor_value_info(
        "Y_h", TensorProto.FLOAT, [num_directions, batch, hidden_size]
    )

    W_initializer = helper.make_tensor(
        "W", TensorProto.FLOAT, W.shape, W.flatten().tolist()
    )
    R_initializer = helper.make_tensor(
        "R", TensorProto.FLOAT, R.shape, R.flatten().tolist()
    )
    B_initializer = helper.make_tensor(
        "B", TensorProto.FLOAT, B.shape, B.flatten().tolist()
    )

    direction = "bidirectional" if bidirectional else "forward"
    gru_node = helper.make_node(
        "GRU",
        inputs=["X", "W", "R", "B", "", "initial_h"],
        outputs=["Y", "Y_h"],
        hidden_size=hidden_size,
        linear_before_reset=linear_before_reset,
        direction=direction,
    )

    graph = helper.make_graph(
        [gru_node],
        "gru_test",
        [input_tensor, initial_h_tensor],
        [output_tensor, output_h_tensor],
        [W_initializer, R_initializer, B_initializer],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx.checker.check_model(model)

    # Run with onnxruntime to get expected output
    ort_session = ort.InferenceSession(model.SerializeToString())
    ort_inputs = {"X": X, "initial_h": initial_h}
    ort_outputs = ort_session.run(None, ort_inputs)
    expected_Y, expected_Y_h = ort_outputs

    # Convert to PyTorch and run
    o2p_gru = ConvertModel(model, experimental=True)
    X_torch = torch.from_numpy(X)
    initial_h_torch = torch.from_numpy(initial_h)

    with torch.no_grad():
        o2p_output, o2p_h_n = o2p_gru(X_torch, initial_h_torch)

    # Compare with onnxruntime outputs
    torch.testing.assert_close(
        o2p_output,
        torch.from_numpy(expected_Y),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        o2p_h_n,
        torch.from_numpy(expected_Y_h),
        rtol=1e-5,
        atol=1e-5,
    )
