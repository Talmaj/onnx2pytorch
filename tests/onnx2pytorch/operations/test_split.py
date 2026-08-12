import numpy as np
import torch
import pytest

from onnx2pytorch.operations import Split
from tests.onnx2pytorch.differential import (
    assert_matches_oracle,
    make_single_node_model,
)


@pytest.fixture
def weight():
    a = torch.rand(15)
    a[[4, 7, 12]] = 0
    return a


@pytest.mark.parametrize("enable_pruning", [True, False])
@pytest.mark.parametrize(
    "split_size_or_sections, number_of_splits", [((5, 5, 5), None), (None, 3)]
)
def test_split(weight, enable_pruning, split_size_or_sections, number_of_splits):
    """keep_size=False"""
    op = Split(
        enable_pruning, split_size_or_sections, number_of_splits, keep_size=False
    )
    s = op(weight)
    assert all(len(x) == 5 for x in s)

    op.set_input_indices((weight,))
    s = op(torch.rand(12))
    assert all(len(x) == 4 for x in s)


@pytest.mark.parametrize("enable_pruning", [True, False])
@pytest.mark.parametrize(
    "split_size_or_sections, number_of_splits", [((5, 5, 5), None), (None, 3)]
)
def test_split_2(weight, enable_pruning, split_size_or_sections, number_of_splits):
    """keep_size=True"""
    op = Split(enable_pruning, split_size_or_sections, number_of_splits, keep_size=True)
    s = op(weight)
    assert all(len(x) == 5 for x in s)

    op.set_input_indices((weight,))
    s = op(torch.rand(12))
    assert all(len(x) == 5 for x in s)

    # keep_size=True expands the input with zeros
    location_of_zeros_in_splits = [4, 2, 2]
    for x, i in zip(s, location_of_zeros_in_splits):
        (idx,) = torch.where(x == 0)
        assert idx == torch.tensor([i])


def test_split_parameter_check(weight):
    with pytest.raises(AssertionError):
        Split(enable_pruning=True, split_size_or_sections=None, number_of_splits=None)


@pytest.mark.parametrize("split_size_or_sections", [(5, 5, 5)])
def test_split_no_enable_pruning(weight, split_size_or_sections):
    op = Split(enable_pruning=False, keep_size=False)
    s = op(weight, split_size_or_sections)
    assert all(len(x) == 5 for x in s)


@pytest.mark.parametrize(
    "size,num_outputs", [(6, 3), (5, 3), (7, 2), (4, 4), (5, 2), (1, 1)]
)
def test_split_num_outputs(size, num_outputs):
    """num_outputs was not extracted at all, and an indivisible dimension gives
    a smaller last chunk rather than an error."""
    x = np.arange(size, dtype=np.float32)
    model = make_single_node_model(
        "Split",
        {"x": x},
        18,
        outputs=tuple("out{}".format(i) for i in range(num_outputs)),
        axis=0,
        num_outputs=num_outputs,
    )
    assert_matches_oracle(model, {"x": x})
