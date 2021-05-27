import numpy as np
import torch
import pytest

from onnx2pytorch.operations.tile import Tile


def test_tile():
    op = Tile()
    x = torch.rand(2, 3, 4, 5)
    repeats = torch.randint(low=1, high=10, size=(x.ndim,))
    z = torch.from_numpy(np.tile(x.numpy(), repeats.numpy()))
    assert torch.equal(op(x, repeats), z)


def test_tile_precomputed():
    op = Tile()
    x = torch.tensor(
        [
            [0, 1],
            [2, 3],
        ],
        dtype=torch.float32,
    )
    repeats = torch.tensor([2, 2], dtype=torch.int64)

    z = torch.tensor(
        [[0, 1, 0, 1], [2, 3, 2, 3], [0, 1, 0, 1], [2, 3, 2, 3]], dtype=torch.float32
    )

    assert torch.equal(op(x, repeats), z)
