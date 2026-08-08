import numpy as np
import pytest
import torch
from onnx import helper, numpy_helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def causal_conv_with_state_reference(
    input, weight, bias=None, past_state=None, activation="none"
):
    """Numpy reference following the ONNX CausalConvWithState specification."""
    batch_size, channels, length = input.shape
    kernel_size = weight.shape[2]

    if past_state is None:
        past_state = np.zeros((batch_size, channels, kernel_size - 1), input.dtype)
    padded = np.concatenate([past_state, input], axis=2)

    output = np.zeros((batch_size, channels, length), dtype=np.float32)
    for b in range(batch_size):
        for c in range(channels):
            for t in range(length):
                window = padded[b, c, t : t + kernel_size]
                output[b, c, t] = np.dot(window, weight[c, 0])
            if bias is not None:
                output[b, c] += bias[c]

    if activation in ("silu", "swish"):
        output = output / (1.0 + np.exp(-output))

    present_state = padded[:, :, padded.shape[2] - (kernel_size - 1) :]
    return output.astype(input.dtype), present_state


def check_causal_conv_with_state(
    input, weight, bias=None, past_state=None, weight_as_initializer=False, **attrs
):
    input_names = ["input", "weight"]
    graph_inputs = [
        helper.make_tensor_value_info("input", TensorProto.FLOAT, list(input.shape)),
    ]
    initializers = []
    feeds = {"input": input}
    if weight_as_initializer:
        initializers.append(numpy_helper.from_array(weight, "weight"))
    else:
        graph_inputs.append(
            helper.make_tensor_value_info(
                "weight", TensorProto.FLOAT, list(weight.shape)
            )
        )
        feeds["weight"] = weight

    for name, value in (("bias", bias), ("past_state", past_state)):
        if value is None:
            input_names.append("")
            continue
        input_names.append(name)
        graph_inputs.append(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, list(value.shape))
        )
        feeds[name] = value

    node = helper.make_node(
        "CausalConvWithState",
        inputs=input_names,
        outputs=["output", "present_state"],
        **attrs,
    )
    graph = helper.make_graph(
        [node],
        "causalconvwithstate_test",
        graph_inputs,
        [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("present_state", TensorProto.FLOAT, None),
        ],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 24)])

    expected = causal_conv_with_state_reference(
        input, weight, bias=bias, past_state=past_state, **attrs
    )

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        res = o2p_model(**{k: torch.from_numpy(v) for k, v in feeds.items()})

    for actual, exp in zip(res, expected):
        np.testing.assert_allclose(actual.numpy(), exp, rtol=1e-5, atol=1e-5)


def make_inputs(batch=2, channels=3, length=5, kernel_size=4, seed=0):
    np.random.seed(seed)
    return (
        np.random.randn(batch, channels, length).astype(np.float32),
        np.random.randn(channels, 1, kernel_size).astype(np.float32),
    )


def test_causal_conv_with_state_basic():
    x, w = make_inputs()
    check_causal_conv_with_state(x, w)


@pytest.mark.parametrize("activation", ["none", "silu", "swish"])
def test_causal_conv_with_state_activation(activation):
    x, w = make_inputs(seed=1)
    check_causal_conv_with_state(x, w, activation=activation)


def test_causal_conv_with_state_bias():
    x, w = make_inputs(seed=2)
    bias = np.random.randn(w.shape[0]).astype(np.float32)
    check_causal_conv_with_state(x, w, bias=bias)


def test_causal_conv_with_state_past_state():
    x, w = make_inputs(seed=3)
    past_state = np.random.randn(x.shape[0], x.shape[1], w.shape[2] - 1).astype(
        np.float32
    )
    check_causal_conv_with_state(x, w, past_state=past_state)


def test_causal_conv_with_state_bias_and_past_state():
    x, w = make_inputs(seed=4)
    bias = np.random.randn(w.shape[0]).astype(np.float32)
    past_state = np.random.randn(x.shape[0], x.shape[1], w.shape[2] - 1).astype(
        np.float32
    )
    check_causal_conv_with_state(
        x, w, bias=bias, past_state=past_state, activation="silu"
    )


def test_causal_conv_with_state_kernel_size_one():
    x, w = make_inputs(kernel_size=1, seed=5)
    check_causal_conv_with_state(x, w)


def test_causal_conv_with_state_weight_initializer():
    x, w = make_inputs(seed=6)
    check_causal_conv_with_state(x, w, weight_as_initializer=True)


def test_causal_conv_with_state_streaming_matches_full_sequence():
    """Feeding chunks while carrying present_state matches one full-sequence call."""
    x, w = make_inputs(length=6, kernel_size=3, seed=7)
    from onnx2pytorch.operations import CausalConvWithState

    op = CausalConvWithState()
    with torch.no_grad():
        full, _ = op(torch.from_numpy(x), torch.from_numpy(w))
        state = None
        chunks = []
        for start in range(0, x.shape[2], 2):
            chunk = torch.from_numpy(x[:, :, start : start + 2])
            out, state = op(chunk, torch.from_numpy(w), None, state)
            chunks.append(out)
    np.testing.assert_allclose(
        torch.cat(chunks, dim=2).numpy(), full.numpy(), rtol=1e-5, atol=1e-6
    )


def test_causal_conv_with_state_unsupported_activation():
    from onnx2pytorch.operations import CausalConvWithState

    with pytest.raises(ValueError, match="activation"):
        CausalConvWithState(activation="relu")
