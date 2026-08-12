import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, numpy_helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def make_sum_body(state_shape, slice_shape, num_scan_outputs=1):
    """Body computing sum_out = sum_in + next, optionally also emitting it."""
    nodes = [helper.make_node("Add", ["sum_in", "next"], ["sum_out"])]
    outputs = [helper.make_tensor_value_info("sum_out", TensorProto.FLOAT, state_shape)]
    for i in range(num_scan_outputs):
        nodes.append(helper.make_node("Identity", ["sum_out"], ["scan_out_%d" % i]))
        outputs.append(
            helper.make_tensor_value_info(
                "scan_out_%d" % i, TensorProto.FLOAT, state_shape
            )
        )
    return helper.make_graph(
        nodes,
        "scan_body",
        [
            helper.make_tensor_value_info("sum_in", TensorProto.FLOAT, state_shape),
            helper.make_tensor_value_info("next", TensorProto.FLOAT, slice_shape),
        ],
        outputs,
    )


def check_scan(body, inputs, output_names, opset_version=16, **attrs):
    node = helper.make_node(
        "Scan",
        inputs=list(inputs.keys()),
        outputs=output_names,
        body=body,
        **attrs,
    )
    graph = helper.make_graph(
        [node],
        "scan_test",
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, list(value.shape))
            for name, value in inputs.items()
        ],
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
            for name in output_names
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset_version)]
    )

    expected = ort.InferenceSession(model.SerializeToString()).run(None, inputs)

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        res = o2p_model(**{k: torch.from_numpy(v) for k, v in inputs.items()})
    if len(output_names) == 1:
        res = [res]

    for actual, exp in zip(res, expected):
        np.testing.assert_allclose(actual.numpy(), exp, rtol=1e-5, atol=1e-5)


def test_scan_sum():
    np.random.seed(0)
    inputs = {
        "init": np.zeros(2, dtype=np.float32),
        "seq": np.random.randn(3, 2).astype(np.float32),
    }
    body = make_sum_body([2], [2])
    check_scan(body, inputs, ["final", "scan"], num_scan_inputs=1)


def test_scan_state_only():
    np.random.seed(1)
    inputs = {
        "init": np.zeros(2, dtype=np.float32),
        "seq": np.random.randn(4, 2).astype(np.float32),
    }
    body = make_sum_body([2], [2], num_scan_outputs=0)
    check_scan(body, inputs, ["final"], num_scan_inputs=1)


def test_scan_multiple_scan_outputs():
    np.random.seed(2)
    inputs = {
        "init": np.zeros(2, dtype=np.float32),
        "seq": np.random.randn(3, 2).astype(np.float32),
    }
    body = make_sum_body([2], [2], num_scan_outputs=2)
    check_scan(body, inputs, ["final", "scan0", "scan1"], num_scan_inputs=1)


@pytest.mark.parametrize("direction", [0, 1])
def test_scan_input_directions(direction):
    np.random.seed(3)
    inputs = {
        "init": np.zeros(2, dtype=np.float32),
        "seq": np.random.randn(4, 2).astype(np.float32),
    }
    body = make_sum_body([2], [2])
    check_scan(
        body,
        inputs,
        ["final", "scan"],
        num_scan_inputs=1,
        scan_input_directions=[direction],
    )


@pytest.mark.parametrize("direction", [0, 1])
def test_scan_output_directions(direction):
    np.random.seed(4)
    inputs = {
        "init": np.zeros(2, dtype=np.float32),
        "seq": np.random.randn(4, 2).astype(np.float32),
    }
    body = make_sum_body([2], [2])
    check_scan(
        body,
        inputs,
        ["final", "scan"],
        num_scan_inputs=1,
        scan_output_directions=[direction],
    )


@pytest.mark.parametrize("axis", [0, 1])
def test_scan_input_axes(axis):
    np.random.seed(5)
    seq = np.random.randn(3, 4).astype(np.float32)
    state_size = seq.shape[1 - axis]
    inputs = {
        "init": np.zeros(state_size, dtype=np.float32),
        "seq": seq,
    }
    body = make_sum_body([state_size], [state_size])
    check_scan(
        body, inputs, ["final", "scan"], num_scan_inputs=1, scan_input_axes=[axis]
    )


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_scan_output_axes(axis):
    np.random.seed(6)
    inputs = {
        "init": np.zeros(2, dtype=np.float32),
        "seq": np.random.randn(3, 2).astype(np.float32),
    }
    body = make_sum_body([2], [2])
    check_scan(
        body, inputs, ["final", "scan"], num_scan_inputs=1, scan_output_axes=[axis]
    )


def test_scan_negative_input_axis():
    np.random.seed(7)
    seq = np.random.randn(2, 5).astype(np.float32)
    inputs = {
        "init": np.zeros(2, dtype=np.float32),
        "seq": seq,
    }
    body = make_sum_body([2], [2])
    check_scan(body, inputs, ["final", "scan"], num_scan_inputs=1, scan_input_axes=[-1])


def test_scan_two_scan_inputs():
    np.random.seed(8)
    body = helper.make_graph(
        [
            helper.make_node("Add", ["a", "b"], ["ab"]),
            helper.make_node("Add", ["sum_in", "ab"], ["sum_out"]),
            helper.make_node("Mul", ["a", "b"], ["prod"]),
        ],
        "scan_body",
        [
            helper.make_tensor_value_info("sum_in", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [2]),
        ],
        [
            helper.make_tensor_value_info("sum_out", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("prod", TensorProto.FLOAT, [2]),
        ],
    )
    inputs = {
        "init": np.zeros(2, dtype=np.float32),
        "a_seq": np.random.randn(3, 2).astype(np.float32),
        "b_seq": np.random.randn(3, 2).astype(np.float32),
    }
    check_scan(body, inputs, ["final", "prods"], num_scan_inputs=2)


def test_scan_two_state_variables():
    np.random.seed(9)
    body = helper.make_graph(
        [
            helper.make_node("Add", ["sum_in", "next"], ["sum_out"]),
            helper.make_node("Mul", ["prod_in", "next"], ["prod_out"]),
            helper.make_node("Add", ["sum_out", "prod_out"], ["both"]),
        ],
        "scan_body",
        [
            helper.make_tensor_value_info("sum_in", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("prod_in", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("next", TensorProto.FLOAT, [2]),
        ],
        [
            helper.make_tensor_value_info("sum_out", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("prod_out", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("both", TensorProto.FLOAT, [2]),
        ],
    )
    inputs = {
        "sum_init": np.zeros(2, dtype=np.float32),
        "prod_init": np.ones(2, dtype=np.float32),
        "seq": np.random.rand(3, 2).astype(np.float32),
    }
    check_scan(body, inputs, ["sum_final", "prod_final", "both"], num_scan_inputs=1)


def test_scan_matmul_state():
    np.random.seed(10)
    body = helper.make_graph(
        [
            helper.make_node("MatMul", ["state_in", "step"], ["state_out"]),
            helper.make_node("Identity", ["state_out"], ["trace"]),
        ],
        "scan_body",
        [
            helper.make_tensor_value_info("state_in", TensorProto.FLOAT, [2, 2]),
            helper.make_tensor_value_info("step", TensorProto.FLOAT, [2, 2]),
        ],
        [
            helper.make_tensor_value_info("state_out", TensorProto.FLOAT, [2, 2]),
            helper.make_tensor_value_info("trace", TensorProto.FLOAT, [2, 2]),
        ],
    )
    inputs = {
        "state_init": np.eye(2, dtype=np.float32),
        "steps": np.random.randn(4, 2, 2).astype(np.float32),
    }
    check_scan(body, inputs, ["state_final", "traces"], num_scan_inputs=1)


def test_scan_with_body_initializer():
    np.random.seed(11)

    offset = np.array([0.5, -0.5], dtype=np.float32)
    body = helper.make_graph(
        [
            helper.make_node("Add", ["next", "offset"], ["shifted"]),
            helper.make_node("Add", ["sum_in", "shifted"], ["sum_out"]),
            helper.make_node("Identity", ["sum_out"], ["scan_out"]),
        ],
        "scan_body",
        [
            helper.make_tensor_value_info("sum_in", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("next", TensorProto.FLOAT, [2]),
        ],
        [
            helper.make_tensor_value_info("sum_out", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("scan_out", TensorProto.FLOAT, [2]),
        ],
        initializer=[numpy_helper.from_array(offset, "offset")],
    )
    inputs = {
        "init": np.zeros(2, dtype=np.float32),
        "seq": np.random.randn(3, 2).astype(np.float32),
    }
    check_scan(body, inputs, ["final", "scan"], num_scan_inputs=1)


def test_scan_reverse_input_and_output():
    np.random.seed(12)
    inputs = {
        "init": np.zeros(2, dtype=np.float32),
        "seq": np.random.randn(4, 2).astype(np.float32),
    }
    body = make_sum_body([2], [2])
    check_scan(
        body,
        inputs,
        ["final", "scan"],
        num_scan_inputs=1,
        scan_input_directions=[1],
        scan_output_directions=[1],
    )
