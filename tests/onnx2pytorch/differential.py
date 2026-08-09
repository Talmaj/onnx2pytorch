"""Helpers to compare a converted single-node model against onnxruntime."""

import numpy as np
import onnx
import pytest
import torch
from onnx import helper, numpy_helper

from onnx2pytorch.convert import ConvertModel

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

NUMPY_TO_TENSOR_PROTO = {
    np.dtype("float32"): onnx.TensorProto.FLOAT,
    np.dtype("float64"): onnx.TensorProto.DOUBLE,
    np.dtype("float16"): onnx.TensorProto.FLOAT16,
    np.dtype("int64"): onnx.TensorProto.INT64,
    np.dtype("int32"): onnx.TensorProto.INT32,
    np.dtype("int16"): onnx.TensorProto.INT16,
    np.dtype("int8"): onnx.TensorProto.INT8,
    np.dtype("uint64"): onnx.TensorProto.UINT64,
    np.dtype("uint32"): onnx.TensorProto.UINT32,
    np.dtype("uint16"): onnx.TensorProto.UINT16,
    np.dtype("uint8"): onnx.TensorProto.UINT8,
    np.dtype("bool"): onnx.TensorProto.BOOL,
}

# Lowest IR version that can express each opset, so that old opsets stay loadable.
_IR_VERSIONS = sorted({(row[2], row[1]) for row in helper.VERSION_TABLE})


class NoOracle(Exception):
    """Raised when neither onnxruntime nor the reference evaluator can run a model."""


def min_ir_version(opset_version):
    for opset, ir_version in _IR_VERSIONS:
        if opset >= opset_version:
            return ir_version
    return _IR_VERSIONS[-1][1]


def make_single_node_model(
    op_type,
    inputs,
    opset_version,
    outputs=("y",),
    initializers=None,
    input_names=None,
    ir_version=None,
    **attributes,
):
    """
    Build a model with a single node at the requested opset.

    Parameters
    ----------
    op_type: str
        ONNX operator type.
    inputs: dict of str to np.ndarray
        Graph inputs, in node input order.
    opset_version: int
        Version of the default ONNX domain to import.
    outputs: sequence of str
        Node output names.
    initializers: dict of str to np.ndarray
        Node inputs provided as initializers instead of graph inputs.
    input_names: sequence of str
        Full node input list, needed when an input is omitted with "".
    """
    initializers = initializers or {}
    if input_names is None:
        input_names = list(inputs) + list(initializers)

    node = helper.make_node(op_type, input_names, list(outputs), **attributes)
    graph = helper.make_graph(
        [node],
        "{}_test".format(op_type.lower()),
        [
            helper.make_tensor_value_info(
                name, NUMPY_TO_TENSOR_PROTO[value.dtype], list(value.shape)
            )
            for name, value in inputs.items()
        ],
        [helper.make_empty_tensor_value_info(name) for name in outputs],
        initializer=[
            numpy_helper.from_array(value, name) for name, value in initializers.items()
        ],
    )
    kwargs = {} if ir_version is None else {"ir_version": ir_version}
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset_version)], **kwargs
    )


def run_onnxruntime(model, inputs):
    session = ort.InferenceSession(model.SerializeToString())
    return session.run(None, {k: v for k, v in inputs.items()})


def run_reference(model, inputs):
    from onnx.reference import ReferenceEvaluator

    return ReferenceEvaluator(model).run(None, {k: v for k, v in inputs.items()})


def run_oracle_strict(model, inputs):
    """Run onnxruntime, falling back to the onnx reference implementation."""
    try:
        return run_onnxruntime(model, inputs)
    except Exception as ort_error:
        try:
            return run_reference(model, inputs)
        except Exception as ref_error:
            raise NoOracle(
                "onnxruntime said {}, the reference evaluator said {}".format(
                    ort_error, ref_error
                )
            )


def run_oracle(model, inputs):
    try:
        return run_oracle_strict(model, inputs)
    except NoOracle as error:
        pytest.skip("No oracle for this case: {}".format(error))


def to_torch(value):
    if value.dtype.kind in ("O", "S", "U"):
        return value
    return torch.from_numpy(value)


def run_converted(model, inputs):
    converted = ConvertModel(model)
    with torch.no_grad():
        outputs = converted(*[to_torch(v) for v in inputs.values()])
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    return [o.numpy() if torch.is_tensor(o) else np.asarray(o) for o in outputs]


def assert_outputs_match(expected, actual, rtol=1e-5, atol=1e-6):
    assert len(actual) == len(expected)
    for exp, act in zip(expected, actual):
        exp = np.asarray(exp)
        if exp.dtype == bool or exp.dtype.kind in ("O", "S", "U", "i", "u"):
            np.testing.assert_array_equal(act, exp)
        else:
            np.testing.assert_allclose(act, exp, rtol=rtol, atol=atol)


def assert_matches_oracle(model, inputs, rtol=1e-5, atol=1e-6):
    expected = run_oracle(model, inputs)
    actual = run_converted(model, inputs)
    assert_outputs_match(expected, actual, rtol=rtol, atol=atol)
    return actual
