import numpy as np
import pytest
import torch
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

from onnx2pytorch.convert import ConvertModel
from onnx2pytorch.operations import LinearAttention


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
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 27)])

    # onnxruntime 1.28 has no LinearAttention kernel
    expected = ReferenceEvaluator(model).run(None, feeds)

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
    with pytest.raises(ValueError, match="update_rule"):
        LinearAttention(q_num_heads=2, kv_num_heads=2, update_rule="unknown")
