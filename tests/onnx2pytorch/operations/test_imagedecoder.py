import io

import numpy as np
import PIL.Image
import pytest
import torch
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

from onnx2pytorch.convert import ConvertModel


def encode(image, image_format):
    buffer = io.BytesIO()
    PIL.Image.fromarray(image).save(buffer, format=image_format)
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8).copy()


def build_model(encoded, **attrs):
    node = helper.make_node("ImageDecoder", inputs=["x"], outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "imagedecoder_test",
        [helper.make_tensor_value_info("x", TensorProto.UINT8, list(encoded.shape))],
        [helper.make_tensor_value_info("y", TensorProto.UINT8, None)],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])


def check_image_decoder(image, image_format, atol=0, **attrs):
    encoded = encode(image, image_format)
    model = build_model(encoded, **attrs)

    # onnxruntime has no ImageDecoder kernel, compare against the onnx reference
    exp_y = ReferenceEvaluator(model).run(None, {"x": encoded})[0]
    with torch.no_grad():
        y = ConvertModel(model)(torch.from_numpy(encoded))

    assert y.dtype == torch.uint8
    assert y.shape == exp_y.shape
    np.testing.assert_allclose(y.numpy().astype(int), exp_y.astype(int), atol=atol)


def make_image(seed=0, height=9, width=7):
    np.random.seed(seed)
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
def test_image_decoder_rgb(image_format):
    check_image_decoder(make_image(), image_format)


@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
def test_image_decoder_bgr(image_format):
    check_image_decoder(make_image(1), image_format, pixel_format="BGR")


@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
def test_image_decoder_grayscale(image_format):
    # the luminance rounding of the decoders differs by at most one level
    check_image_decoder(make_image(2), image_format, atol=1, pixel_format="Grayscale")


def test_image_decoder_larger_image():
    check_image_decoder(make_image(3, 64, 32), "PNG")


def test_image_decoder_unsupported_pixel_format():
    encoded = encode(make_image(), "PNG")
    model = build_model(encoded, pixel_format="YCbCr")
    with pytest.raises(NotImplementedError):
        ConvertModel(model)
