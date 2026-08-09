import numpy as np
import torch
import pytest
from onnx.backend.test.case.node.pad import pad_impl

from onnx2pytorch.operations import Pad
from onnx2pytorch.utils import extract_padding_params
from tests.onnx2pytorch.differential import (
    assert_matches_oracle,
    make_single_node_model,
)


@pytest.fixture
def inp():
    return torch.rand(1, 3, 10, 10)


# ONNX convention: [begin_d0, ..., begin_dN, end_d0, ..., end_dN].
# The asymmetric cases are what distinguish a correct begin/end ordering from a
# swapped one; symmetric pads give the same shape either way.
ONNX_PADS = [
    ([0, 0, 0, 1, 0, 0, 0, 1], [1, 3, 10, 12]),
    ([0, 0, 2, 1, 0, 0, 2, 1], [1, 3, 14, 12]),
    ([4, 3, 2, 1, 4, 3, 2, 1], [9, 9, 14, 12]),
    ([0, 0, 1, 2, 0, 0, 30, 40], [1, 3, 41, 52]),
    ([1, 0, 3, 0, 0, 2, 0, 4], [2, 5, 13, 14]),
]


@pytest.mark.parametrize("onnx_pads, new_shape", ONNX_PADS)
def test_pad_static(inp, onnx_pads, new_shape):
    """Padding passed at initialization is pre-converted by the converter."""
    op = Pad(padding=extract_padding_params(onnx_pads))
    out = op(inp)

    expected = pad_impl(inp.numpy(), np.array(onnx_pads), "constant", 0)
    assert list(out.shape) == new_shape
    assert np.array_equal(out.numpy(), expected)


@pytest.mark.parametrize("onnx_pads, new_shape", ONNX_PADS)
def test_pad_dynamic(inp, onnx_pads, new_shape):
    """Padding passed to forward arrives in raw ONNX convention."""
    op = Pad()
    out = op(inp, onnx_pads)

    expected = pad_impl(inp.numpy(), np.array(onnx_pads), "constant", 0)
    assert list(out.shape) == new_shape
    assert np.array_equal(out.numpy(), expected)


@pytest.mark.parametrize("onnx_pads, new_shape", ONNX_PADS)
def test_pad_dynamic_accepts_tensor(inp, onnx_pads, new_shape):
    """ONNX supplies pads as an int64 tensor, not a python list."""
    op = Pad()
    out = op(inp, torch.tensor(onnx_pads, dtype=torch.int64))
    assert list(out.shape) == new_shape


def test_pad_dynamic_constant_value(inp):
    """constant_value arrives as a 0-d tensor, which F.pad does not accept."""
    onnx_pads = [0, 0, 1, 2, 0, 0, 3, 4]
    op = Pad()
    out = op(inp, onnx_pads, torch.tensor(2.5))

    expected = pad_impl(inp.numpy(), np.array(onnx_pads), "constant", 2.5)
    assert np.allclose(out.numpy(), expected)


def test_pad_raise_error(inp):
    op = Pad()

    # padding should be passed either in init or forward
    with pytest.raises(TypeError):
        op(inp)


PAD_MODES = ["constant", "reflect", "edge", "wrap"]


@pytest.mark.parametrize("mode", PAD_MODES)
@pytest.mark.parametrize("opset_version", [11, 13, 18, 19])
def test_pad_mode_with_pads_input(opset_version, mode):
    """ONNX mode names have to be mapped onto torch's, and edge used to raise."""
    np.random.seed(0)
    x = np.random.randn(1, 2, 4, 5).astype(np.float32)
    pads = np.array([0, 0, 1, 2, 0, 0, 3, 1], dtype=np.int64)
    model = make_single_node_model(
        "Pad",
        {"x": x},
        opset_version,
        input_names=["x", "pads"],
        initializers={"pads": pads},
        mode=mode,
    )
    assert_matches_oracle(model, {"x": x})


@pytest.mark.parametrize("mode", PAD_MODES)
def test_pad_mode_with_pads_attribute(mode):
    np.random.seed(0)
    x = np.random.randn(1, 2, 4, 5).astype(np.float32)
    model = make_single_node_model(
        "Pad", {"x": x}, 2, pads=[0, 0, 1, 2, 0, 0, 3, 1], mode=mode
    )
    assert_matches_oracle(model, {"x": x})


def test_pad_value_attribute():
    """Pad-2's value attribute collided with the global value mapping."""
    np.random.seed(0)
    x = np.random.randn(1, 2, 4, 5).astype(np.float32)
    model = make_single_node_model(
        "Pad", {"x": x}, 2, pads=[0, 0, 1, 2, 0, 0, 3, 1], value=2.5
    )
    assert_matches_oracle(model, {"x": x})


def test_pad_paddings_attribute():
    """Opset 1 spells the pads attribute paddings."""
    np.random.seed(0)
    x = np.random.randn(1, 2, 4, 5).astype(np.float32)
    model = make_single_node_model(
        "Pad", {"x": x}, 1, paddings=[0, 0, 1, 2, 0, 0, 3, 1], value=2.5
    )
    assert_matches_oracle(model, {"x": x})


@pytest.mark.parametrize("opset_version", [2, 11, 18])
def test_pad_batch_and_channel_pads(opset_version):
    np.random.seed(0)
    x = np.random.randn(1, 2, 4, 5).astype(np.float32)
    onnx_pads = [1, 1, 1, 2, 1, 1, 3, 1]
    if opset_version == 2:
        model = make_single_node_model("Pad", {"x": x}, 2, pads=onnx_pads)
    else:
        model = make_single_node_model(
            "Pad",
            {"x": x},
            opset_version,
            input_names=["x", "pads"],
            initializers={"pads": np.array(onnx_pads, dtype=np.int64)},
        )
    assert_matches_oracle(model, {"x": x})
