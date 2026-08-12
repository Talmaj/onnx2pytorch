"""Cases where the converter has to fail loudly rather than diverge silently."""

import inspect
import re

import numpy as np
import onnx
import pytest
from onnx import defs, helper, TensorProto

from onnx2pytorch.convert import ConvertModel
from onnx2pytorch.convert.operations import convert_operations
from tests.onnx2pytorch.differential import (
    assert_no_runtime_oracle,
    assert_outputs_match,
    make_single_node_model,
    run_converted,
    run_oracle,
)


def test_scan_opset_8_is_rejected():
    """Scan-8 prepends sequence_lens and adds a batch dimension everywhere."""
    body = helper.make_graph(
        [helper.make_node("Identity", ["state_in"], ["state_out"])],
        "scan_body",
        [
            helper.make_tensor_value_info("state_in", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("scan_in", TensorProto.FLOAT, [2]),
        ],
        [helper.make_tensor_value_info("state_out", TensorProto.FLOAT, [2])],
    )
    node = helper.make_node(
        "Scan",
        ["", "state", "values"],
        ["final_state"],
        body=body,
        num_scan_inputs=1,
    )
    graph = helper.make_graph(
        [node],
        "scan_test",
        [
            helper.make_tensor_value_info("state", TensorProto.FLOAT, [1, 2]),
            helper.make_tensor_value_info("values", TensorProto.FLOAT, [1, 3, 2]),
        ],
        [helper.make_empty_tensor_value_info("final_state")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 8)])

    with pytest.raises(NotImplementedError, match="Scan at opset 8"):
        ConvertModel(model)


def test_batch_normalization_non_spatial_is_rejected():
    x = np.random.randn(1, 2, 3, 3).astype(np.float32)
    per_element = np.ones((2, 3, 3), dtype=np.float32)
    model = make_single_node_model(
        "BatchNormalization",
        {"x": x},
        8,
        initializers={
            "scale": per_element,
            "bias": np.zeros_like(per_element),
            "mean": np.zeros_like(per_element),
            "var": np.ones_like(per_element),
        },
        spatial=0,
    )
    with pytest.raises(NotImplementedError, match="spatial=0"):
        ConvertModel(model)


def test_batch_normalization_non_spatial_per_channel_is_accepted():
    """spatial=0 next to 1D parameters is per channel, so it must not be rejected.

    resnet18v1 from the model zoo sets spatial=0 on all 20 nodes while shipping
    per-channel parameters, which only onnxruntime's Conv+BatchNormalization
    fusion lets it run at all. With per-channel parameters the attribute cannot
    mean anything else, so the rejection has to key off the parameter rank.
    """
    np.random.seed(0)
    x = np.random.randn(2, 3, 4, 4).astype(np.float32)
    params = {
        "scale": np.random.randn(3).astype(np.float32),
        "bias": np.random.randn(3).astype(np.float32),
        "mean": np.random.randn(3).astype(np.float32),
        "var": np.abs(np.random.randn(3).astype(np.float32)) + 0.1,
    }
    model = make_single_node_model(
        "BatchNormalization", {"x": x}, 8, initializers=params, spatial=0
    )
    # Both runtimes reject the unfused node, so spatial=1 has to stand in
    assert_no_runtime_oracle(model, {"x": x})
    spatial = make_single_node_model(
        "BatchNormalization", {"x": x}, 8, initializers=params, spatial=1
    )
    assert_outputs_match(run_oracle(spatial, {"x": x}), run_converted(model, {"x": x}))


def test_pad_with_unknown_mode_is_rejected():
    x = np.random.randn(1, 2, 4, 4).astype(np.float32)
    model = make_single_node_model(
        "Pad",
        {"x": x},
        18,
        input_names=["x", "pads"],
        initializers={"pads": np.zeros(8, dtype=np.int64)},
        mode="something_else",
    )
    with pytest.raises(NotImplementedError, match="Pad mode"):
        ConvertModel(model)


def test_no_onnx_operator_relies_on_the_torch_name_fallback():
    """convert_operations ends in getattr(torch, op_type.lower()), which guesses.

    Every ai.onnx operator must be dispatched explicitly instead, so that a new
    onnx release cannot silently start resolving an operator by name alone.
    """
    source = inspect.getsource(convert_operations)
    dispatched = set(re.findall(r'node\.op_type == "([A-Za-z0-9_]+)"', source))
    schemas = {
        s.name
        for s in defs.get_all_schemas_with_history()
        if s.domain in ("", "ai.onnx")
    }
    assert sorted(schemas - dispatched) == []
