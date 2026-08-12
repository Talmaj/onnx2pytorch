import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def build_model(past_cache, update, write_indices=None, **attrs):
    input_names = ["past_cache", "update"]
    value_infos = [
        helper.make_tensor_value_info(
            "past_cache", TensorProto.FLOAT, list(past_cache.shape)
        ),
        helper.make_tensor_value_info("update", TensorProto.FLOAT, list(update.shape)),
    ]
    feed = {"past_cache": past_cache, "update": update}
    if write_indices is not None:
        input_names.append("write_indices")
        value_infos.append(
            helper.make_tensor_value_info(
                "write_indices", TensorProto.INT64, list(write_indices.shape)
            )
        )
        feed["write_indices"] = write_indices

    node = helper.make_node(
        "TensorScatter", inputs=input_names, outputs=["present_cache"], **attrs
    )
    graph = helper.make_graph(
        [node],
        "tensorscatter_test",
        value_infos,
        [helper.make_tensor_value_info("present_cache", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 24)])
    return model, feed


def check_tensor_scatter(past_cache, update, write_indices=None, **attrs):
    model, feed = build_model(past_cache, update, write_indices, **attrs)

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp = ort_session.run(None, feed)[0]

    with torch.no_grad():
        y = ConvertModel(model)(*[torch.from_numpy(v) for v in feed.values()])
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


def test_tensor_scatter_circular_batch_exceeds_max_sequence_length():
    """A batch larger than max_sequence_length, where the spec pseudocode misleads.

    The prose defines circular mode as "the update index is modulo
    max_sequence_length", but the pseudocode writes
    ``cache_idx = tuple(np.mod(np.asarray(cache_idx), max_sequence_length))``,
    which also wraps the batch and head indices in the prefix. That only shows up
    once a prefix dimension exceeds max_sequence_length, as here, and it would
    make the operator scatter into the wrong sample. onnxruntime wraps the
    sequence position alone, which is what the prose describes.
    """
    past_cache = make_cache((7, 2, 3, 3), 14)
    update = make_cache((7, 2, 2, 3), 15)
    write_indices = np.array([2, 1, 0, 2, 1, 0, 2], dtype=np.int64)
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
    model, _ = build_model(past_cache, update, mode="wraparound")
    with pytest.raises(NotImplementedError):
        ConvertModel(model)
