import io
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from onnx2pytorch.convert import ConvertModel


def test_train_multiple_models():
    original_model = torch.nn.Linear(in_features=3, out_features=5)
    dummy_input = torch.randn(4, 3)
    original_output = original_model(dummy_input)

    bitstream = io.BytesIO()
    torch.onnx.export(original_model, dummy_input, bitstream, opset_version=11)

    bitstream.seek(0)
    onnx_model = onnx.load(bitstream)
    o2p_model = ConvertModel(onnx_model, experimental=True)
    o2p_model2 = ConvertModel(onnx_model, experimental=True)
    o2p_output = o2p_model(dummy_input)
    assert torch.equal(original_output, o2p_output)

    onnx_model_serial = onnx_model.SerializeToString()
    with torch.no_grad():
        for name, param in o2p_model.named_parameters():
            param.copy_(torch.zeros(*param.shape))
    assert onnx_model.SerializeToString() == onnx_model_serial

    o2p_output_after = o2p_model(dummy_input)
    assert not torch.equal(original_output, o2p_output_after)
    o2p_output2_after = o2p_model2(dummy_input)
    assert torch.equal(original_output, o2p_output2_after)
