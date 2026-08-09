import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


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

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_y = ort_session.run(None, dict(zip(input_names, feeds)))[0]

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


@pytest.mark.parametrize("interleaved", [0, 1])
def test_rotary_embedding_partial_rotation_3d(interleaved):
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
        interleaved=interleaved,
        rotary_embedding_dim=6,
    )
