import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel
from tests.onnx2pytorch.differential import (
    assert_matches_oracle,
    assert_no_runtime_oracle,
    make_single_node_model,
)


def build_attention_model(inputs, num_outputs=1, **attrs):
    input_names = []
    graph_inputs = []
    feeds = {}
    for name, value in inputs.items():
        if value is None:
            input_names.append("")
            continue
        elem_type = (
            TensorProto.BOOL
            if value.dtype == np.bool_
            else (TensorProto.INT64 if value.dtype == np.int64 else TensorProto.FLOAT)
        )
        input_names.append(name)
        graph_inputs.append(
            helper.make_tensor_value_info(name, elem_type, list(value.shape))
        )
        feeds[name] = value

    output_names = ["Y", "present_key", "present_value", "qk_matmul_output"]
    output_names = output_names[:num_outputs]
    node = helper.make_node(
        "Attention", inputs=input_names, outputs=output_names, **attrs
    )
    graph = helper.make_graph(
        [node],
        "attention_test",
        graph_inputs,
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
            for name in output_names
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 24)])
    return model, feeds


def run_o2p(model, feeds, num_outputs):
    o2p_model = ConvertModel(model)
    with torch.no_grad():
        res = o2p_model(**{k: torch.from_numpy(v) for k, v in feeds.items()})
    return [res] if num_outputs == 1 else list(res)


def check_attention(inputs, num_outputs=1, **attrs):
    model, feeds = build_attention_model(inputs, num_outputs, **attrs)

    ort_session = ort.InferenceSession(model.SerializeToString())
    expected = ort_session.run(None, feeds)

    for actual, exp in zip(run_o2p(model, feeds, num_outputs), expected):
        np.testing.assert_allclose(actual.numpy(), exp, rtol=1e-4, atol=1e-5)


def make_qkv_4d(batch=2, q_heads=3, kv_heads=3, q_len=4, kv_len=5, head_size=8, seed=0):
    np.random.seed(seed)
    return dict(
        Q=np.random.randn(batch, q_heads, q_len, head_size).astype(np.float32),
        K=np.random.randn(batch, kv_heads, kv_len, head_size).astype(np.float32),
        V=np.random.randn(batch, kv_heads, kv_len, head_size).astype(np.float32),
    )


def test_attention_4d_basic():
    check_attention(make_qkv_4d())


def test_attention_3d_basic():
    np.random.seed(1)
    inputs = dict(
        Q=np.random.randn(2, 4, 12).astype(np.float32),
        K=np.random.randn(2, 5, 12).astype(np.float32),
        V=np.random.randn(2, 5, 12).astype(np.float32),
    )
    check_attention(inputs, q_num_heads=3, kv_num_heads=3)


def test_attention_3d_different_v_head_size():
    np.random.seed(2)
    inputs = dict(
        Q=np.random.randn(2, 4, 12).astype(np.float32),
        K=np.random.randn(2, 5, 12).astype(np.float32),
        V=np.random.randn(2, 5, 6).astype(np.float32),
    )
    check_attention(inputs, q_num_heads=3, kv_num_heads=3)


def test_attention_gqa_4d():
    check_attention(make_qkv_4d(q_heads=6, kv_heads=2, seed=3))


def test_attention_gqa_3d():
    np.random.seed(4)
    inputs = dict(
        Q=np.random.randn(2, 4, 24).astype(np.float32),
        K=np.random.randn(2, 5, 8).astype(np.float32),
        V=np.random.randn(2, 5, 8).astype(np.float32),
    )
    check_attention(inputs, q_num_heads=6, kv_num_heads=2)


def test_attention_is_causal():
    check_attention(make_qkv_4d(q_len=5, kv_len=5, seed=5), is_causal=1)


def test_attention_scale():
    check_attention(make_qkv_4d(seed=6), scale=0.25)


def test_attention_bool_mask():
    np.random.seed(7)
    inputs = make_qkv_4d(seed=7)
    inputs["attn_mask"] = np.random.rand(4, 5) > 0.4
    check_attention(inputs)


def test_attention_float_mask():
    inputs = make_qkv_4d(seed=8)
    np.random.seed(8)
    inputs["attn_mask"] = np.random.randn(2, 3, 4, 5).astype(np.float32)
    check_attention(inputs)


def test_attention_fully_masked_row():
    inputs = make_qkv_4d(seed=9)
    mask = np.ones((4, 5), dtype=bool)
    mask[1] = False
    inputs["attn_mask"] = mask
    check_attention(inputs)


def test_attention_past_key_value():
    inputs = make_qkv_4d(q_len=2, kv_len=3, seed=10)
    np.random.seed(10)
    inputs["attn_mask"] = None
    inputs["past_key"] = np.random.randn(2, 3, 4, 8).astype(np.float32)
    inputs["past_value"] = np.random.randn(2, 3, 4, 8).astype(np.float32)
    check_attention(inputs, num_outputs=3)


def test_attention_past_key_value_causal():
    inputs = make_qkv_4d(q_len=2, kv_len=2, seed=11)
    np.random.seed(11)
    inputs["attn_mask"] = None
    inputs["past_key"] = np.random.randn(2, 3, 4, 8).astype(np.float32)
    inputs["past_value"] = np.random.randn(2, 3, 4, 8).astype(np.float32)
    check_attention(inputs, num_outputs=3, is_causal=1)


@pytest.mark.parametrize("dtype", [np.bool_, np.float32])
def test_attention_short_mask_is_padded(dtype):
    """A mask shorter than the kv sequence, which no runtime will run directly.

    The spec says the last dimension of attn_mask "can also be shorter than
    total_sequence_length and will be padded to total_sequence_length with
    negative infinity", but onnxruntime rejects the short form outright and
    onnx's reference evaluator cannot broadcast it together with
    nonpad_kv_seqlen. The oracle is therefore onnxruntime on the mask the spec
    says the short one stands for, which is the padding rule itself rather than
    an independent derivation of it.
    """
    inputs = make_qkv_4d(q_len=3, kv_len=6, seed=12)
    np.random.seed(12)
    mask = np.random.rand(3, 4) > 0.3
    pad_value = False if dtype == np.bool_ else -np.inf
    if dtype != np.bool_:
        mask = np.where(mask, 0, -np.inf).astype(np.float32)
    inputs["attn_mask"] = mask
    inputs["past_key"] = None
    inputs["past_value"] = None
    inputs["nonpad_kv_seqlen"] = np.array([4, 3], dtype=np.int64)

    padded = dict(inputs)
    padded["attn_mask"] = np.pad(mask, ((0, 0), (0, 2)), constant_values=pad_value)
    model, feeds = build_attention_model(padded)
    expected = ort.InferenceSession(model.SerializeToString()).run(None, feeds)

    short_model, short_feeds = build_attention_model(inputs)
    assert_no_runtime_oracle(short_model, short_feeds)
    for actual, exp in zip(run_o2p(short_model, short_feeds, 1), expected):
        np.testing.assert_allclose(actual.numpy(), exp, rtol=1e-4, atol=1e-5)


def test_attention_softcap():
    check_attention(make_qkv_4d(seed=13), softcap=2.0)


@pytest.mark.parametrize("qk_matmul_output_mode", [0, 1, 2, 3])
def test_attention_qk_matmul_output_mode(qk_matmul_output_mode):
    inputs = make_qkv_4d(seed=14)
    np.random.seed(14)
    inputs["attn_mask"] = np.random.randn(4, 5).astype(np.float32)
    check_attention(inputs, num_outputs=4, qk_matmul_output_mode=qk_matmul_output_mode)


@pytest.mark.parametrize("qk_matmul_output_mode", [0, 1, 2, 3])
def test_attention_qk_matmul_output_mode_with_softcap(qk_matmul_output_mode):
    """Only mode 1 and up see the softcap, mode 0 is the bare matmul.

    onnx's reference evaluator returns the softcapped scores for mode 0 too,
    which contradicts the spec and onnxruntime, hence the explicit onnxruntime
    oracle here.
    """
    check_attention(
        make_qkv_4d(seed=15),
        num_outputs=4,
        softcap=3.0,
        qk_matmul_output_mode=qk_matmul_output_mode,
    )


def test_attention_nonpad_kv_seqlen():
    inputs = make_qkv_4d(q_len=3, kv_len=6, seed=16)
    inputs["attn_mask"] = None
    inputs["past_key"] = None
    inputs["past_value"] = None
    inputs["nonpad_kv_seqlen"] = np.array([6, 4], dtype=np.int64)
    check_attention(inputs, num_outputs=3)


def test_attention_nonpad_kv_seqlen_causal():
    inputs = make_qkv_4d(q_len=3, kv_len=6, seed=17)
    inputs["attn_mask"] = None
    inputs["past_key"] = None
    inputs["past_value"] = None
    inputs["nonpad_kv_seqlen"] = np.array([6, 5], dtype=np.int64)
    check_attention(inputs, num_outputs=4, is_causal=1)


@pytest.mark.parametrize("softmax_precision", [1, 10])
@pytest.mark.parametrize("num_outputs", [1, 4])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_attention_softmax_precision(softmax_precision, num_outputs, dtype):
    """The softmax result stayed in the requested precision, so the following
    matmul met a float32 tensor where the value was float16. The fast path
    softmaxes in the input type, so it cannot serve this attribute at all.

    A float16 softmax rounds differently than onnxruntime's, hence the tolerance.
    """
    inputs = {name: value.astype(dtype) for name, value in make_qkv_4d().items()}
    model = make_single_node_model(
        "Attention",
        inputs,
        24,
        outputs=("Y", "present_key", "present_value", "qk_matmul_output")[:num_outputs],
        softmax_precision=softmax_precision,
    )
    tolerance = 1e-2 if np.dtype(dtype) == np.float16 else 5e-3
    assert_matches_oracle(model, inputs, rtol=tolerance, atol=tolerance)
