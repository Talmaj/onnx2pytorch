import io
import onnx
import pytest
import torch

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
def test_single_layer_lstm(
    bidirectional, input_size, hidden_size, seq_len, batch, test_seq_len, test_batch
):
    torch.manual_seed(42)
    num_layers = 1
    num_directions = bidirectional + 1
    lstm = torch.nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )
    input = torch.randn(seq_len, batch, input_size)
    h_0 = torch.randn(num_layers * num_directions, batch, hidden_size)
    c_0 = torch.randn(num_layers * num_directions, batch, hidden_size)
    output, (h_n, c_n) = lstm(input, (h_0, c_0))
    bitstream = io.BytesIO()
    torch.onnx.export(
        model=lstm,
        args=(input, (h_0, c_0)),
        f=bitstream,
        input_names=["input", "h_0", "c_0"],
        opset_version=11,
        dynamic_axes={
            "input": {0: "seq_len", 1: "batch"},
            "h_0": {1: "batch"},
            "c_0": {1: "batch"},
        },
    )
    bitstream_data = bitstream.getvalue()

    onnx_lstm = onnx.ModelProto.FromString(bitstream_data)
    o2p_lstm = ConvertModel(onnx_lstm, experimental=True)
    with torch.no_grad():
        o2p_output, o2p_h_n, o2p_c_n = o2p_lstm(input, h_0, c_0)
        assert torch.equal(o2p_output, output)
        assert torch.equal(o2p_h_n, h_n)
        assert torch.equal(o2p_c_n, c_n)

    onnx_lstm = onnx.ModelProto.FromString(bitstream_data)
    o2p_lstm = ConvertModel(onnx_lstm, experimental=True)
    with torch.no_grad():
        o2p_output, o2p_h_n, o2p_c_n = o2p_lstm(h_0=h_0, input=input, c_0=c_0)
        assert torch.equal(o2p_output, output)
        assert torch.equal(o2p_h_n, h_n)
        assert torch.equal(o2p_c_n, c_n)
        with pytest.raises(KeyError):
            o2p_output, o2p_h_n, o2p_c_n = o2p_lstm(h_0=h_0, input=input)
        with pytest.raises(Exception):
            # Even though initial states are optional for nn.LSTM(),
            # we adhere to onnxruntime convention that inputs are provided
            # as either all positional or all keyword arguments.
            o2p_output, o2p_h_n, o2p_c_n = o2p_lstm(input, h_0=h_0, c_0=c_0)
