import numpy as np
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def rotary_embedding_reference(
    input,
    cos_cache,
    sin_cache,
    position_ids=None,
    interleaved=0,
    rotary_embedding_dim=0,
    num_heads=0,
):
    """Numpy reference following the ONNX RotaryEmbedding specification."""
    original_shape = input.shape
    if input.ndim == 4:
        input = np.transpose(input, (0, 2, 1, 3))
    batch_size, sequence_length = input.shape[0], input.shape[1]
    if input.ndim == 3:
        head_size = input.shape[2] // num_heads
        input = np.reshape(input, [batch_size, sequence_length, num_heads, head_size])
    head_size = input.shape[3]

    if not rotary_embedding_dim:
        rotary_embedding_dim = head_size
    x_rotate = input[:, :, :, :rotary_embedding_dim]
    x_not_rotate = input[:, :, :, rotary_embedding_dim:]

    if position_ids is not None:
        cos_cache = cos_cache[position_ids]
        sin_cache = sin_cache[position_ids]
    cos_cache = np.expand_dims(cos_cache, axis=2)
    sin_cache = np.expand_dims(sin_cache, axis=2)

    if interleaved:
        x1 = x_rotate[:, :, :, 0::2]
        x2 = x_rotate[:, :, :, 1::2]
    else:
        x1, x2 = np.split(x_rotate, 2, axis=-1)

    real = (cos_cache * x1) - (sin_cache * x2)
    imag = (sin_cache * x1) + (cos_cache * x2)

    if interleaved:
        real = np.expand_dims(real, axis=-1)
        imag = np.expand_dims(imag, axis=-1)
        x_rotate = np.reshape(np.concatenate((real, imag), axis=-1), x_rotate.shape)
    else:
        x_rotate = np.concatenate((real, imag), axis=-1)

    output = np.concatenate((x_rotate, x_not_rotate), axis=-1)
    if len(original_shape) == 3:
        return np.reshape(output, original_shape)
    return np.transpose(output, (0, 2, 1, 3))


def check_rotary_embedding(x, cos_cache, sin_cache, position_ids=None, **attrs):
    input_names = ["x", "cos_cache", "sin_cache"]
    inputs = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape)),
        helper.make_tensor_value_info(
            "cos_cache", TensorProto.FLOAT, list(cos_cache.shape)
        ),
        helper.make_tensor_value_info(
            "sin_cache", TensorProto.FLOAT, list(sin_cache.shape)
        ),
    ]
    feeds = [x, cos_cache, sin_cache]
    if position_ids is not None:
        input_names.append("position_ids")
        inputs.append(
            helper.make_tensor_value_info(
                "position_ids", TensorProto.INT64, list(position_ids.shape)
            )
        )
        feeds.append(position_ids)

    node = helper.make_node(
        "RotaryEmbedding", inputs=input_names, outputs=["y"], **attrs
    )
    graph = helper.make_graph(
        [node],
        "rotaryembedding_test",
        inputs,
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])

    exp_y = rotary_embedding_reference(
        x, cos_cache, sin_cache, position_ids=position_ids, **attrs
    )

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(*[torch.from_numpy(f) for f in feeds])

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-5, atol=1e-5)


def make_caches(max_position, half_dim):
    positions = np.arange(max_position, dtype=np.float32).reshape(-1, 1)
    freqs = 1.0 / (10000 ** (np.arange(half_dim, dtype=np.float32) / half_dim))
    angles = positions * freqs
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


@pytest.mark.parametrize("interleaved", [0, 1])
def test_rotary_embedding_4d(interleaved):
    np.random.seed(0)
    x = np.random.randn(2, 3, 4, 8).astype(np.float32)
    cos_cache, sin_cache = make_caches(4, 4)
    cos_cache = np.broadcast_to(cos_cache, (2, 4, 4)).copy()
    sin_cache = np.broadcast_to(sin_cache, (2, 4, 4)).copy()
    check_rotary_embedding(x, cos_cache, sin_cache, interleaved=interleaved)


@pytest.mark.parametrize("interleaved", [0, 1])
def test_rotary_embedding_3d(interleaved):
    np.random.seed(1)
    x = np.random.randn(2, 4, 24).astype(np.float32)
    cos_cache, sin_cache = make_caches(4, 4)
    cos_cache = np.broadcast_to(cos_cache, (2, 4, 4)).copy()
    sin_cache = np.broadcast_to(sin_cache, (2, 4, 4)).copy()
    check_rotary_embedding(
        x, cos_cache, sin_cache, interleaved=interleaved, num_heads=3
    )


def test_rotary_embedding_position_ids():
    np.random.seed(2)
    x = np.random.randn(2, 3, 4, 8).astype(np.float32)
    cos_cache, sin_cache = make_caches(16, 4)
    position_ids = np.array([[0, 1, 2, 3], [4, 6, 8, 10]], dtype=np.int64)
    check_rotary_embedding(x, cos_cache, sin_cache, position_ids=position_ids)


@pytest.mark.parametrize("interleaved", [0, 1])
def test_rotary_embedding_partial_rotation(interleaved):
    np.random.seed(3)
    x = np.random.randn(2, 3, 4, 8).astype(np.float32)
    cos_cache, sin_cache = make_caches(16, 2)
    position_ids = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int64)
    check_rotary_embedding(
        x,
        cos_cache,
        sin_cache,
        position_ids=position_ids,
        interleaved=interleaved,
        rotary_embedding_dim=4,
    )


def test_rotary_embedding_partial_rotation_3d():
    np.random.seed(4)
    x = np.random.randn(2, 4, 24).astype(np.float32)
    cos_cache, sin_cache = make_caches(16, 3)
    position_ids = np.array([[0, 2, 4, 6], [1, 3, 5, 7]], dtype=np.int64)
    check_rotary_embedding(
        x,
        cos_cache,
        sin_cache,
        position_ids=position_ids,
        num_heads=3,
        rotary_embedding_dim=6,
    )
