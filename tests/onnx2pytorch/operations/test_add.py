import pytest
import torch

from onnx2pytorch.operations.add import Add


@pytest.fixture(
    params=[[[10, 20, 30], list(range(32))], [[10, 20, 30], [8, 9, 10, 11, 12]]]
)
def input_indices(request):
    return request.param


@pytest.fixture
def updated_input_indices(input_indices):
    """Re-index indices after removal of channels that contain only zeros."""
    s = list(set(input_indices[0]).union(input_indices[1]))
    new = [[s.index(x) for x in indices] for indices in input_indices]
    return new


@pytest.fixture
def input_shape(input_indices):
    n = len(set(input_indices[0]).union(input_indices[1]))
    return (32, n, 8, 8)


@pytest.fixture
def original_shape():
    return (32, 32, 8, 8)


@pytest.fixture
def inputs(input_indices):
    """Pruned smaller inputs."""
    return [
        torch.ones(32, len(input_indices[0]), 8, 8),
        2 * torch.ones(32, len(input_indices[1]), 8, 8),
    ]


def test_add(input_shape, inputs, updated_input_indices):
    """Test addition of differently sized inputs."""
    op = Add(input_shape, updated_input_indices)
    out = op(*inputs)

    for i in range(32):
        if i in updated_input_indices[0] and i in updated_input_indices[1]:
            assert out[0, i, 0, 0] == 3
        elif i in updated_input_indices[0]:
            assert out[0, i, 0, 0] == 1
        elif i in updated_input_indices[1]:
            assert out[0, i, 0, 0] == 2


def test_simple_add():
    a = torch.zeros(2, 2)
    b = torch.zeros(2, 2)
    b[0] = 1
    op = Add()
    out = op(a, b)
    assert torch.allclose(out, torch.tensor([[1.0, 1], [0, 0]]))


def test_add_2(input_shape, inputs, updated_input_indices):
    """Test addition of differently sized inputs."""
    inputs_2 = [torch.zeros(*input_shape), torch.zeros(*input_shape)]
    for i in range(2):
        inp = inputs_2[i]
        idx = updated_input_indices[i]
        inp[:, idx] = i + 1

    op = Add()
    out_true = op(*inputs_2)

    op = Add(input_shape, updated_input_indices)
    out = op(*inputs)

    assert torch.allclose(out_true, out, atol=1e-7), "Outputs differ."


@pytest.mark.parametrize(
    "inputs",
    [
        [torch.tensor(1), torch.ones(10, 10)],
        [torch.ones(10, 10), torch.tensor(1)],
        [torch.tensor(1), torch.ones(10, 10), torch.tensor(0)],
    ],
)
def test_add_constant(inputs):
    op = Add()
    out = op(*inputs)
    assert torch.equal(out, 2 * torch.ones(10, 10))


def test_set_input_indices(
    input_indices, updated_input_indices, input_shape, original_shape
):
    inputs = [torch.zeros(*original_shape), torch.zeros(*original_shape)]
    for inp, idx in zip(inputs, input_indices):
        inp[:, idx] = 1

    op = Add()
    op.set_input_indices(inputs)
    assert op.input_shape == input_shape
    for true_idx, calc_idx in zip(updated_input_indices, op.input_indices):
        assert true_idx == calc_idx.tolist()


@pytest.mark.parametrize("inp", [torch.triu(torch.ones(10, 10), 1), torch.tensor(1)])
def test_set_input_indices_triu(inp):
    act = torch.ones(10, 10)
    inputs = [act, inp]
    op = Add()
    op.set_input_indices(inputs)
    assert op.input_indices is None
