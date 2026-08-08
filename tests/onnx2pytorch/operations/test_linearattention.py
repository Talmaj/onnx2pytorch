import numpy as np
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def unpack_heads_reference(x, num_heads):
    batch_size, sequence_length, hidden_size = x.shape
    x = x.reshape(batch_size, sequence_length, num_heads, hidden_size // num_heads)
    return x.transpose(0, 2, 1, 3)


def linear_attention_reference(
    query,
    key,
    value,
    past_state=None,
    decay=None,
    beta=None,
    q_num_heads=None,
    kv_num_heads=None,
    scale=None,
    update_rule="gated_delta",
    chunk_size=None,
):
    """Numpy reference following the ONNX LinearAttention specification."""
    gating = update_rule in ("gated", "gated_delta")
    delta_correction = update_rule in ("delta", "gated_delta")

    batch_size, sequence_length, _ = query.shape
    key_dim = query.shape[-1] // q_num_heads
    value_dim = value.shape[-1] // kv_num_heads
    group_size = q_num_heads // kv_num_heads

    q4 = unpack_heads_reference(query, q_num_heads).astype(np.float32)
    k4 = unpack_heads_reference(key, kv_num_heads).astype(np.float32)
    v4 = unpack_heads_reference(value, kv_num_heads).astype(np.float32)

    if gating:
        if decay.shape[-1] == kv_num_heads:
            decay4 = decay.reshape(
                batch_size, sequence_length, kv_num_heads, 1
            ).transpose(0, 2, 1, 3)
        else:
            decay4 = unpack_heads_reference(decay, kv_num_heads)
        decay4 = decay4.astype(np.float32)
    if delta_correction:
        beta4 = beta.reshape(batch_size, sequence_length, beta.shape[-1], 1).transpose(
            0, 2, 1, 3
        )
        beta4 = beta4.astype(np.float32)

    if past_state is None:
        state_dtype = query.dtype
        state = np.zeros(
            (batch_size, kv_num_heads, key_dim, value_dim), dtype=np.float32
        )
    else:
        state_dtype = past_state.dtype
        state = past_state.astype(np.float32).copy()

    scale_val = 1.0 / np.sqrt(key_dim) if not scale else float(scale)

    outputs = np.zeros(
        (batch_size, q_num_heads, sequence_length, value_dim), dtype=np.float32
    )
    for i in range(sequence_length):
        q_t, k_t, v_t = q4[:, :, i, :], k4[:, :, i, :], v4[:, :, i, :]
        if gating:
            state = state * np.exp(decay4[:, :, i, :])[..., None]
        if delta_correction:
            retrieved = np.einsum("bhdm,bhd->bhm", state, k_t)
            v_t = beta4[:, :, i, :] * (v_t - retrieved)
        state = state + k_t[..., :, None] * v_t[..., None, :]
        read_state = state if group_size == 1 else np.repeat(state, group_size, axis=1)
        outputs[:, :, i, :] = scale_val * np.einsum("bhd,bhdm->bhm", q_t, read_state)

    output = outputs.transpose(0, 2, 1, 3).reshape(
        batch_size, sequence_length, q_num_heads * value_dim
    )
    return output.astype(query.dtype), state.astype(state_dtype)


def check_linear_attention(inputs, **attrs):
    input_names = []
    graph_inputs = []
    feeds = {}
    for name, value in inputs.items():
        if value is None:
            input_names.append("")
            continue
        input_names.append(name)
        graph_inputs.append(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, list(value.shape))
        )
        feeds[name] = value

    node = helper.make_node(
        "LinearAttention",
        inputs=input_names,
        outputs=["output", "present_state"],
        **attrs,
    )
    graph = helper.make_graph(
        [node],
        "linearattention_test",
        graph_inputs,
        [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("present_state", TensorProto.FLOAT, None),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 24)])

    expected = linear_attention_reference(**inputs, **attrs)

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        res = o2p_model(**{k: torch.from_numpy(v) for k, v in feeds.items()})

    for actual, exp in zip(res, expected):
        np.testing.assert_allclose(actual.numpy(), exp, rtol=1e-4, atol=1e-5)


def make_inputs(
    update_rule,
    q_num_heads=2,
    kv_num_heads=2,
    key_dim=4,
    value_dim=4,
    batch=2,
    seq=3,
    with_past_state=False,
    decay_per_key_dim=False,
    beta_scalar=False,
    seed=0,
):
    np.random.seed(seed)
    inputs = dict(
        query=np.random.randn(batch, seq, q_num_heads * key_dim).astype(np.float32),
        key=np.random.randn(batch, seq, kv_num_heads * key_dim).astype(np.float32),
        value=np.random.randn(batch, seq, kv_num_heads * value_dim).astype(np.float32),
    )
    if with_past_state:
        inputs["past_state"] = np.random.randn(
            batch, kv_num_heads, key_dim, value_dim
        ).astype(np.float32)
    else:
        inputs["past_state"] = None
    if update_rule in ("gated", "gated_delta"):
        last = kv_num_heads * key_dim if decay_per_key_dim else kv_num_heads
        inputs["decay"] = (-np.random.rand(batch, seq, last)).astype(np.float32)
    else:
        inputs["decay"] = None
    if update_rule in ("delta", "gated_delta"):
        last = 1 if beta_scalar else kv_num_heads
        inputs["beta"] = np.random.rand(batch, seq, last).astype(np.float32)
    else:
        inputs["beta"] = None
    return inputs


@pytest.mark.parametrize("update_rule", ["linear", "gated", "delta", "gated_delta"])
def test_linear_attention_update_rules(update_rule):
    inputs = make_inputs(update_rule, seed=1)
    check_linear_attention(
        inputs, q_num_heads=2, kv_num_heads=2, update_rule=update_rule
    )


def test_linear_attention_default_update_rule():
    inputs = make_inputs("gated_delta", seed=2)
    check_linear_attention(inputs, q_num_heads=2, kv_num_heads=2)


def test_linear_attention_group_query():
    inputs = make_inputs("gated_delta", q_num_heads=6, kv_num_heads=2, seed=3)
    check_linear_attention(inputs, q_num_heads=6, kv_num_heads=2)


def test_linear_attention_past_state():
    inputs = make_inputs("gated_delta", with_past_state=True, seed=4)
    check_linear_attention(inputs, q_num_heads=2, kv_num_heads=2)


def test_linear_attention_decay_per_key_dim():
    inputs = make_inputs("gated", decay_per_key_dim=True, seed=5)
    check_linear_attention(inputs, q_num_heads=2, kv_num_heads=2, update_rule="gated")


def test_linear_attention_beta_scalar():
    inputs = make_inputs("delta", beta_scalar=True, seed=6)
    check_linear_attention(inputs, q_num_heads=2, kv_num_heads=2, update_rule="delta")


def test_linear_attention_scale():
    inputs = make_inputs("linear", seed=7)
    check_linear_attention(
        inputs, q_num_heads=2, kv_num_heads=2, update_rule="linear", scale=0.5
    )


def test_linear_attention_different_value_dim():
    inputs = make_inputs("gated_delta", value_dim=6, seed=8)
    check_linear_attention(inputs, q_num_heads=2, kv_num_heads=2)


def test_linear_attention_unsupported_update_rule():
    from onnx2pytorch.operations import LinearAttention

    with pytest.raises(ValueError, match="update_rule"):
        LinearAttention(q_num_heads=2, kv_num_heads=2, update_rule="unknown")
