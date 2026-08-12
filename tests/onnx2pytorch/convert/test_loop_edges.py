"""Loop cases around its optional inputs, an empty body run, and nested graphs."""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from tests.onnx2pytorch.differential import assert_matches_oracle


def value(name, shape, dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, shape)


def counting_body(scan_output=True, stop_at=None, nodes=None):
    """A body that adds one to its carried value, and reports it as a scan output.

    With stop_at the body ends the loop itself rather than passing the incoming
    condition through, which is the only way to end a loop without a trip count.
    """
    body_nodes = list(nodes or [helper.make_node("Add", ["acc", "one"], ["acc_out"])])
    if scan_output:
        body_nodes.append(helper.make_node("Identity", ["acc_out"], ["scan_out"]))
    if stop_at is None:
        body_nodes.append(helper.make_node("Identity", ["cond_in"], ["cond_out"]))
    else:
        body_nodes.append(
            helper.make_node("Less", ["iteration", "limit"], ["cond_out"])
        )

    outputs = [value("cond_out", [], TensorProto.BOOL), value("acc_out", [1])]
    if scan_output:
        outputs.append(value("scan_out", [1]))
    return helper.make_graph(
        body_nodes,
        "body",
        [
            value("iteration", [], TensorProto.INT64),
            value("cond_in", [], TensorProto.BOOL),
            value("acc", [1]),
        ],
        outputs,
        initializer=[
            numpy_helper.from_array(np.array([1.0], dtype=np.float32), "one"),
            numpy_helper.from_array(
                np.array(stop_at if stop_at is not None else 0, dtype=np.int64), "limit"
            ),
        ],
    )


def loop_model(body, trip_count=3, condition=True, omit=(), opset_version=16):
    scan_output = len(body.output) > 2
    node_inputs = [
        "" if "M" in omit else "M",
        "" if "cond" in omit else "cond",
        "acc_init",
    ]
    outputs = ["acc_final"] + (["scan_all"] if scan_output else [])
    node = helper.make_node("Loop", node_inputs, outputs, body=body)
    graph = helper.make_graph(
        [node],
        "test",
        [value("acc_init", [1])],
        [helper.make_empty_tensor_value_info(name) for name in outputs],
        initializer=[
            numpy_helper.from_array(np.array(trip_count, dtype=np.int64), "M"),
            numpy_helper.from_array(np.array(condition), "cond"),
        ],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset_version)]
    )


@pytest.mark.parametrize("scan_output", [True, False])
@pytest.mark.parametrize(
    "trip_count,condition",
    [(3, True), (0, True), (3, False), (0, False), (1, True)],
)
def test_loop_trip_count_and_condition(scan_output, trip_count, condition):
    """A body that never ran left the carried outputs unset, and there was
    nothing to concatenate for the scan outputs."""
    model = loop_model(
        counting_body(scan_output), trip_count=trip_count, condition=condition
    )
    assert_matches_oracle(model, {"acc_init": np.array([0.0], dtype=np.float32)})


def test_loop_without_a_trip_count():
    """An omitted M is unbounded, it used to be compared against as a None."""
    model = loop_model(counting_body(stop_at=2), omit=("M",))
    assert_matches_oracle(model, {"acc_init": np.array([0.0], dtype=np.float32)})


def test_loop_without_an_initial_condition():
    """An omitted cond starts out true, it used to be falsy and stop the loop."""
    model = loop_model(counting_body(), omit=("cond",))
    assert_matches_oracle(model, {"acc_init": np.array([0.0], dtype=np.float32)})


def test_loop_without_either_optional_input():
    model = loop_model(counting_body(stop_at=2), omit=("M", "cond"))
    assert_matches_oracle(model, {"acc_init": np.array([0.0], dtype=np.float32)})


def test_loop_containing_an_if():
    """Only a nested Loop used to be given the enclosing scope, so a nested If
    ran without one."""
    branch_nodes = {
        "then": helper.make_node("Add", ["acc", "one"], ["result"]),
        "else": helper.make_node("Sub", ["acc", "one"], ["result"]),
    }
    branches = {
        name: helper.make_graph(
            [node],
            name,
            [],
            [value("result", [1])],
            initializer=[
                numpy_helper.from_array(np.array([1.0], dtype=np.float32), "one")
            ],
        )
        for name, node in branch_nodes.items()
    }
    body = counting_body(
        scan_output=False,
        nodes=[
            helper.make_node("Less", ["iteration", "limit"], ["pick"]),
            helper.make_node(
                "If",
                ["pick"],
                ["acc_out"],
                then_branch=branches["then"],
                else_branch=branches["else"],
            ),
        ],
    )
    body.initializer.append(
        numpy_helper.from_array(np.array(2, dtype=np.int64), "limit")
    )
    model = loop_model(body, trip_count=4)
    assert_matches_oracle(model, {"acc_init": np.array([0.0], dtype=np.float32)})


def test_loop_containing_a_multi_output_node():
    """Every output of a nested node was stored under the name of the first."""
    body = helper.make_graph(
        [
            helper.make_node("Split", ["acc"], ["low", "high"], axis=0, num_outputs=2),
            helper.make_node("Add", ["low", "high"], ["sum"]),
            helper.make_node("Concat", ["sum", "low"], ["acc_out"], axis=0),
            helper.make_node("Identity", ["cond_in"], ["cond_out"]),
        ],
        "body",
        [
            value("iteration", [], TensorProto.INT64),
            value("cond_in", [], TensorProto.BOOL),
            value("acc", [2]),
        ],
        [value("cond_out", [], TensorProto.BOOL), value("acc_out", [2])],
    )
    node = helper.make_node("Loop", ["M", "cond", "acc_init"], ["acc_final"], body=body)
    graph = helper.make_graph(
        [node],
        "test",
        [value("acc_init", [2])],
        [helper.make_empty_tensor_value_info("acc_final")],
        initializer=[
            numpy_helper.from_array(np.array(2, dtype=np.int64), "M"),
            numpy_helper.from_array(np.array(True), "cond"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    assert_matches_oracle(model, {"acc_init": np.array([1.0, 3.0], dtype=np.float32)})
