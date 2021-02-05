import torch
import pytest

from onnx2pytorch.convert.debug import debug_model_conversion
from onnx2pytorch.helpers import to_onnx


@pytest.fixture
def inp_size():
    return (1, 3, 10, 10)


@pytest.fixture
def inp(inp_size):
    return torch.rand(*inp_size)


@pytest.fixture
def model():
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 10, 3, 1, 1), torch.nn.Conv2d(10, 3, 3, 1, 1)
    )


@pytest.fixture
def onnx_model(model, inp_size):
    return to_onnx(model, inp_size)


def test_debug_model_conversion(model, onnx_model, inp):
    pred_act = model[0](inp)
    debug_model_conversion(onnx_model, [inp], pred_act, onnx_model.graph.node[0])


def test_debug_model_conversion_raise_error(model, onnx_model, inp):
    model.eval()
    pred_act = torch.rand(1, 10, 10, 10)

    with pytest.raises(AssertionError):
        debug_model_conversion(onnx_model, [inp], pred_act, onnx_model.graph.node[0])

    with pytest.raises(TypeError):
        debug_model_conversion(onnx_model, inp, pred_act, onnx_model.graph.node[0])
