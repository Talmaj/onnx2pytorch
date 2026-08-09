"""Cases where the converter has to fail loudly rather than diverge silently."""

import numpy as np
import onnx
import pytest
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel
from tests.onnx2pytorch.differential import make_single_node_model


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
