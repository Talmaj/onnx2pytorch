import numpy as np
import pytest

from tests.onnx2pytorch.differential import (
    assert_matches_oracle,
    make_single_node_model,
)


@pytest.mark.parametrize("end", [None, 0, 1, 3, -1, 9])
@pytest.mark.parametrize("start", [None, 0, 1, 2, -1, -2, 5, -9])
def test_shape_start_end(start, end):
    """start and end are honoured from opset 15 on, and clamped to the rank."""
    np.random.seed(0)
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    attributes = {}
    if start is not None:
        attributes["start"] = start
    if end is not None:
        attributes["end"] = end
    model = make_single_node_model("Shape", {"x": x}, 21, **attributes)
    assert_matches_oracle(model, {"x": x})


@pytest.mark.parametrize("opset_version", [1, 13, 15, 21])
def test_shape_full_rank(opset_version):
    np.random.seed(0)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    model = make_single_node_model("Shape", {"x": x}, opset_version)
    assert_matches_oracle(model, {"x": x})


@pytest.mark.parametrize("start,end", [(1, 1), (2, 2), (0, 0), (-1, -1)])
def test_shape_empty_slice_is_still_int64(start, end):
    """An empty slice used to build a float32 tensor, since it had no values to
    infer the type from."""
    np.random.seed(0)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    model = make_single_node_model("Shape", {"x": x}, 21, start=start, end=end)
    assert_matches_oracle(model, {"x": x})
