import numpy as np
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def tensor_scatter_reference(
    past_cache, update, write_indices=None, axis=-2, mode="linear"
):
    """Numpy reference following the pseudocode of the ONNX TensorScatter specification."""
    axis = axis % past_cache.ndim
    max_sequence_length = past_cache.shape[axis]
    sequence_length = update.shape[axis]
    if write_indices is None:
        write_indices = np.zeros(past_cache.shape[0], dtype=np.int64)

    present_cache = past_cache.copy()
    for prefix_idx in np.ndindex(past_cache.shape[:axis]):
        batch_idx = prefix_idx[0]
        for sequence_idx in range(sequence_length):
            cache_position = write_indices[batch_idx] + sequence_idx
            if mode == "circular":
                cache_position = cache_position % max_sequence_length
            present_cache[(*prefix_idx, cache_position)] = update[
                (*prefix_idx, sequence_idx)
            ]
    return present_cache


def build_model(past_cache, update, write_indices=None, **attrs):
    inputs = ["past_cache", "update"]
    value_infos = [
        helper.make_tensor_value_info(
            "past_cache", TensorProto.FLOAT, list(past_cache.shape)
        ),
        helper.make_tensor_value_info("update", TensorProto.FLOAT, list(update.shape)),
    ]
    feed = {"past_cache": past_cache, "update": update}
    if write_indices is not None:
        inputs.append("write_indices")
        value_infos.append(
            helper.make_tensor_value_info(
                "write_indices", TensorProto.INT64, list(write_indices.shape)
            )
        )
        feed["write_indices"] = write_indices

    node = helper.make_node(
        "TensorScatter", inputs=inputs, outputs=["present_cache"], **attrs
    )
    graph = helper.make_graph(
        [node],
        "tensorscatter_test",
        value_infos,
        [helper.make_tensor_value_info("present_cache", TensorProto.FLOAT, None)],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 24)])


def check_tensor_scatter(past_cache, update, write_indices=None, **attrs):
    model = build_model(past_cache, update, write_indices, **attrs)

    # TensorScatter is opset 24, neither onnxruntime nor onnx can run it here
    exp = tensor_scatter_reference(past_cache, update, write_indices, **attrs)
    feed = [past_cache, update] + ([] if write_indices is None else [write_indices])
    with torch.no_grad():
        y = ConvertModel(model)(*[torch.from_numpy(v) for v in feed])
    np.testing.assert_array_equal(y.numpy(), exp)


def make_cache(shape, seed):
    np.random.seed(seed)
    return np.random.randn(*shape).astype(np.float32)


def test_tensor_scatter_prefill():
    past_cache = make_cache((2, 3, 8, 4), 0)
    update = make_cache((2, 3, 8, 4), 1)
    check_tensor_scatter(past_cache, update)


def test_tensor_scatter_partial_prefill():
    past_cache = make_cache((2, 3, 8, 4), 2)
    update = make_cache((2, 3, 5, 4), 3)
    check_tensor_scatter(past_cache, update)


def test_tensor_scatter_decode():
    past_cache = make_cache((3, 2, 8, 4), 4)
    update = make_cache((3, 2, 1, 4), 5)
    write_indices = np.array([0, 3, 7], dtype=np.int64)
    check_tensor_scatter(past_cache, update, write_indices)


def test_tensor_scatter_write_indices_per_batch():
    past_cache = make_cache((4, 2, 6, 3), 6)
    update = make_cache((4, 2, 2, 3), 7)
    write_indices = np.array([0, 1, 2, 4], dtype=np.int64)
    check_tensor_scatter(past_cache, update, write_indices)


def test_tensor_scatter_circular():
    past_cache = make_cache((2, 2, 5, 3), 8)
    update = make_cache((2, 2, 3, 3), 9)
    write_indices = np.array([3, 4], dtype=np.int64)
    check_tensor_scatter(past_cache, update, write_indices, mode="circular")


@pytest.mark.parametrize("axis", [1, 2, -2, -3])
def test_tensor_scatter_axis(axis):
    past_cache = make_cache((2, 6, 5, 3), 10)
    shape = list(past_cache.shape)
    shape[axis] = 2
    update = make_cache(tuple(shape), 11)
    write_indices = np.array([0, 1], dtype=np.int64)
    check_tensor_scatter(past_cache, update, write_indices, axis=axis)


def test_tensor_scatter_unsupported_mode():
    past_cache = make_cache((2, 2, 4, 3), 12)
    update = make_cache((2, 2, 1, 3), 13)
    model = build_model(past_cache, update, mode="wraparound")
    with pytest.raises(NotImplementedError):
        ConvertModel(model)
