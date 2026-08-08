import numpy as np
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def softmax_reference(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    safe_max = np.where(np.isneginf(x_max), 0, x_max)
    tmp = np.exp(x - safe_max)
    s = np.sum(tmp, axis=axis, keepdims=True)
    return tmp / np.where(s == 0, 1, s)


def causal_bias_reference(base, offset, q_length, kv_length):
    i = np.arange(q_length).reshape(q_length, 1)
    j = np.arange(kv_length).reshape(1, kv_length)
    per_batch = np.ndim(offset) > 0
    if per_batch:
        allowed = j <= (i + np.reshape(offset, (-1, 1, 1)))
    else:
        allowed = j <= (i + int(offset))
    causal = np.where(allowed, base.dtype.type(0), base.dtype.type(-np.inf))
    if per_batch:
        base = base.reshape((1,) * (4 - base.ndim) + base.shape)
        return base + causal.reshape(-1, 1, q_length, kv_length)
    return base + causal


def attention_reference(
    Q,
    K,
    V,
    attn_mask=None,
    past_key=None,
    past_value=None,
    nonpad_kv_seqlen=None,
    scale=None,
    is_causal=0,
    q_num_heads=None,
    kv_num_heads=None,
    softcap=None,
    qk_matmul_output_mode=0,
):
    """Numpy reference following the ONNX Attention specification."""
    input_rank = Q.ndim
    batch_size = Q.shape[0]
    if input_rank == 3:

        def to_4d(x, num_heads):
            head_size = x.shape[2] // num_heads
            x = np.reshape(x, [batch_size, x.shape[1], num_heads, head_size])
            return np.transpose(x, (0, 2, 1, 3))

        Q = to_4d(Q, q_num_heads)
        K = to_4d(K, kv_num_heads)
        V = to_4d(V, kv_num_heads)

    if scale is None:
        scale = 1 / np.sqrt(Q.shape[3])
    root_scale = np.sqrt(scale)

    present_key = K if past_key is None else np.concatenate((past_key, K), axis=2)
    present_value = V if past_value is None else np.concatenate((past_value, V), axis=2)
    K, V = present_key, present_value

    q_length, kv_length = Q.shape[2], K.shape[2]
    attn_bias = np.zeros((q_length, kv_length), dtype=Q.dtype)

    if attn_mask is not None:
        pad_width = kv_length - attn_mask.shape[-1]
        if pad_width > 0:
            pad_shape = [(0, 0)] * (attn_mask.ndim - 1) + [(0, pad_width)]
            pad_value = False if attn_mask.dtype == np.bool_ else -np.inf
            attn_mask = np.pad(attn_mask, pad_shape, constant_values=pad_value)
        if attn_mask.dtype == np.bool_:
            attn_mask = np.where(attn_mask, Q.dtype.type(0), Q.dtype.type(-np.inf))

    if is_causal:
        base = attn_bias if attn_mask is None else attn_mask.copy()
        if past_key is None and nonpad_kv_seqlen is not None:
            offset = nonpad_kv_seqlen.reshape(-1) - q_length
        else:
            offset = 0 if past_key is None else past_key.shape[2]
        attn_bias = causal_bias_reference(base, offset, q_length, kv_length)
    elif attn_mask is not None:
        attn_bias = attn_bias + attn_mask

    if nonpad_kv_seqlen is not None:
        attn_bias = attn_bias.reshape((1,) * (4 - attn_bias.ndim) + attn_bias.shape)
        padding_mask = np.arange(kv_length) < nonpad_kv_seqlen[:, np.newaxis]
        padding_mask = padding_mask.reshape(batch_size, 1, 1, kv_length)
        attn_bias = attn_bias + np.where(padding_mask, 0, -np.inf)

    heads_q = Q.shape[1] if q_num_heads is None else q_num_heads
    heads_kv = K.shape[1] if kv_num_heads is None else kv_num_heads
    if heads_q != heads_kv and heads_q % heads_kv == 0:
        repeats = heads_q // heads_kv
        K = np.repeat(K, repeats, axis=1)
        V = np.repeat(V, repeats, axis=1)

    qk = np.matmul(Q * root_scale, np.transpose(K, (0, 1, 3, 2)) * root_scale)
    if softcap:
        qk = np.tanh(qk / softcap) * softcap
    qk_with_bias = qk + attn_bias
    qk_matmul_output = qk_with_bias if qk_matmul_output_mode == 2 else qk

    probs = softmax_reference(qk_with_bias)
    row_all_masked = np.isneginf(np.max(attn_bias, axis=-1, keepdims=True))
    probs = np.where(row_all_masked, 0, probs)
    if qk_matmul_output_mode == 3:
        qk_matmul_output = probs

    y = np.matmul(probs, V).astype(Q.dtype)
    if input_rank == 3:
        y = np.transpose(y, (0, 2, 1, 3))
        y = np.reshape(y, (batch_size, q_length, -1))
    return y, present_key, present_value, qk_matmul_output.astype(Q.dtype)


def check_attention(inputs, num_outputs=1, **attrs):
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

    expected = attention_reference(**inputs, **attrs)[:num_outputs]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        res = o2p_model(**{k: torch.from_numpy(v) for k, v in feeds.items()})
    if num_outputs == 1:
        res = [res]

    for actual, exp in zip(res, expected):
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


def test_attention_short_mask_is_padded():
    inputs = make_qkv_4d(q_len=2, kv_len=2, seed=12)
    np.random.seed(12)
    inputs["attn_mask"] = np.random.rand(2, 2) > 0.3
    inputs["past_key"] = np.random.randn(2, 3, 3, 8).astype(np.float32)
    inputs["past_value"] = np.random.randn(2, 3, 3, 8).astype(np.float32)
    check_attention(inputs, num_outputs=3)


def test_attention_softcap():
    check_attention(make_qkv_4d(seed=13), softcap=2.0)


@pytest.mark.parametrize("qk_matmul_output_mode", [0, 1, 2, 3])
def test_attention_qk_matmul_output_mode(qk_matmul_output_mode):
    inputs = make_qkv_4d(seed=14)
    np.random.seed(14)
    inputs["attn_mask"] = np.random.randn(4, 5).astype(np.float32)
    check_attention(inputs, num_outputs=4, qk_matmul_output_mode=qk_matmul_output_mode)


def test_attention_qk_matmul_output_mode_with_softcap():
    check_attention(
        make_qkv_4d(seed=15), num_outputs=4, softcap=3.0, qk_matmul_output_mode=1
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
