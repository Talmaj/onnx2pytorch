import pytest
import torch
import onnx
import numpy as np
from onnx import numpy_helper
from onnx.backend.test.case.node.gemm import gemm_reference_implementation
from torch import nn

from onnx2pytorch.convert import convert_linear_layer
from onnx2pytorch.helpers import to_converted


@pytest.fixture
def embedding():
    # inp (35, 1)
    return nn.Embedding(28785, 200)


@pytest.fixture
def encoder():
    # inp (35, 1, 200)
    encoder_layers = nn.TransformerEncoderLayer(200, 2, 100, 0.2)
    model = nn.TransformerEncoder(encoder_layers, 2)
    model.eval()
    return model


@pytest.mark.skip("Not passing tox test")
def test_convert(encoder):
    inp = torch.rand(35, 1, 200).to(torch.float32)
    mask = torch.ones(35, 35).to(torch.float32)
    with torch.no_grad():
        out_true = encoder(inp, mask)

    converted_model = to_converted(encoder, ((35, 1, 200), (35, 35)))
    converted_model.batch_dim = 1

    out = converted_model(inp, mask)
    assert torch.allclose(out_true, out, atol=1e-6)


def test_convert_linear_layer_trasB1():
    node = onnx.helper.make_node(
        "Gemm", inputs=["a", "b", "c"], outputs=["y"], transB=1
    )
    a = np.random.ranf([3, 6]).astype(np.float32)
    b = np.random.ranf([4, 6]).astype(np.float32)
    c = np.zeros([1, 4]).astype(np.float32)
    y = gemm_reference_implementation(a, b, c, transB=1)

    params = [numpy_helper.from_array(b), numpy_helper.from_array(c)]
    op = convert_linear_layer(node, params)
    op.eval()
    out = op(torch.from_numpy(a))
    torch.allclose(torch.from_numpy(y), out)


def test_convert_linear_layer_default():
    node = onnx.helper.make_node("Gemm", inputs=["a", "b", "c"], outputs=["y"])
    a = np.random.ranf([3, 6]).astype(np.float32)
    b = np.random.ranf([6, 4]).astype(np.float32)
    c = np.random.ranf([3, 4]).astype(np.float32)
    y = gemm_reference_implementation(a, b, c)

    params = [numpy_helper.from_array(b), numpy_helper.from_array(c)]
    op = convert_linear_layer(node, params)
    op.eval()
    out = op(torch.from_numpy(a))
    torch.allclose(torch.from_numpy(y), out)


def test_convert_linear_layer_transB0():
    node = onnx.helper.make_node(
        "Gemm", inputs=["a", "b", "c"], outputs=["y"], transB=0
    )
    a = np.random.ranf([3, 6]).astype(np.float32)
    b = np.random.ranf([6, 4]).astype(np.float32)
    c = np.random.ranf([3, 4]).astype(np.float32)
    y = gemm_reference_implementation(a, b, c, transB=0)

    params = [numpy_helper.from_array(b), numpy_helper.from_array(c)]
    op = convert_linear_layer(node, params)
    op.eval()
    out = op(torch.from_numpy(a))
    torch.allclose(torch.from_numpy(y), out)


def test_convert_linear_layer_alpha():
    node = onnx.helper.make_node(
        "Gemm", inputs=["a", "b", "c"], outputs=["y"], alpha=0.5
    )
    a = np.random.ranf([3, 5]).astype(np.float32)
    b = np.random.ranf([5, 4]).astype(np.float32)
    c = np.zeros([1, 4]).astype(np.float32)
    y = gemm_reference_implementation(a, b, c, alpha=0.5)

    params = [numpy_helper.from_array(b), numpy_helper.from_array(c)]
    op = convert_linear_layer(node, params)
    op.eval()
    out = op(torch.from_numpy(a))
    torch.allclose(torch.from_numpy(y), out)


def test_convert_linear_layer_all():
    node = onnx.helper.make_node(
        "Gemm",
        inputs=["a", "b", "c"],
        outputs=["y"],
        alpha=0.25,
        beta=0.35,
        transA=0,
        transB=1,
    )
    a = np.random.ranf([4, 3]).astype(np.float32).transpose()
    b = np.random.ranf([5, 4]).astype(np.float32)
    c = np.random.ranf([1, 5]).astype(np.float32)
    y = gemm_reference_implementation(
        a, b, c, transA=0, transB=1, alpha=0.25, beta=0.35
    )

    params = [numpy_helper.from_array(b), numpy_helper.from_array(c)]
    op = convert_linear_layer(node, params)
    op.eval()
    out = op(torch.from_numpy(a))
    torch.allclose(torch.from_numpy(y), out)
