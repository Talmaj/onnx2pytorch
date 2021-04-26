import os
import io

import onnx
import numpy as np
import torch
import onnxruntime as ort

from onnx2pytorch import convert


def test_onnx2pytorch(onnx_model, onnx_model_outputs, onnx_inputs):
    model = convert.ConvertModel(onnx_model)
    model.eval()
    model.cpu()
    with torch.no_grad():
        outputs = model(*(torch.from_numpy(i) for i in onnx_inputs.values()))

    if not isinstance(outputs, list):
        outputs = [outputs]

    outputs = [x.cpu().numpy() for x in outputs]

    for output, onnx_model_output in zip(outputs, onnx_model_outputs):
        print("mse", ((onnx_model_output - output) ** 2).sum() / onnx_model_output.size)
        np.testing.assert_allclose(onnx_model_output, output, atol=1e-5, rtol=1e-3)


def test_onnx2pytorch2onnx(onnx_model, onnx_model_outputs, onnx_inputs):
    """Test that conversion works both ways."""
    torch_inputs = [torch.from_numpy(x) for x in onnx_inputs.values()]

    model = convert.ConvertModel(onnx_model)
    model.eval()
    model.cpu()

    bitstream = io.BytesIO()
    torch.onnx.export(
        model,
        tuple(torch_inputs),
        bitstream,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=list(onnx_inputs.keys()),
    )

    # for some reason the following check fails the circleci with segmentation fault
    if not os.environ.get("CIRCLECI"):
        onnx_model = onnx.ModelProto.FromString(bitstream.getvalue())
        onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession(bitstream.getvalue())
    outputs = ort_session.run(None, onnx_inputs)

    for output, onnx_model_output in zip(outputs, onnx_model_outputs):
        print("mse", ((onnx_model_output - output) ** 2).sum() / onnx_model_output.size)
        np.testing.assert_allclose(onnx_model_output, output, atol=1e-5, rtol=1e-3)
