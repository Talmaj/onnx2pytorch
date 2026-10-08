"""Turn Constant nodes into initializers so weight-bearing ops can read them.

Exporters, and the branches of If and Loop, often produce a convolution weight
with a Constant node instead of a graph initializer. Folding those tensors (and
Constant -> Identity chains) into initializers lets the existing converters
build a real nn.Conv / nn.Linear / BatchNorm instead of seeing a missing
parameter. Constants defined in an outer graph are passed in and stay visible
to subgraphs.
"""

import numpy as np
from onnx import numpy_helper


def _attribute(node, name):
    for attr in node.attribute:
        if attr.name == name:
            return attr
    return None


def constant_node_to_array(node):
    """The numeric value of a Constant node, or None if it cannot be folded."""
    if node.op_type != "Constant":
        return None
    try:
        value = _attribute(node, "value")
        if value is not None:
            return numpy_helper.to_array(value.t)
        sparse = _attribute(node, "sparse_value")
        if sparse is not None:
            return numpy_helper.to_array(sparse.sparse_tensor)
        value_float = _attribute(node, "value_float")
        if value_float is not None:
            return np.array(value_float.f, dtype=np.float32)
        value_floats = _attribute(node, "value_floats")
        if value_floats is not None:
            return np.asarray(value_floats.floats, dtype=np.float32)
        value_int = _attribute(node, "value_int")
        if value_int is not None:
            return np.array(value_int.i, dtype=np.int64)
        value_ints = _attribute(node, "value_ints")
        if value_ints is not None:
            return np.asarray(value_ints.ints, dtype=np.int64)
    except Exception:
        return None
    return None


def _contiguous(array):
    """ascontiguousarray promotes a 0-d value to shape (1,); keep scalars scalar."""
    array = np.asarray(array)
    if array.ndim == 0:
        return array
    return np.ascontiguousarray(array)


def _as_initializer(array, name):
    """Encode a folded value. numpy_helper turns a 0-d array into shape (1,)."""
    array = _contiguous(array)
    if array.ndim == 0:
        tensor = numpy_helper.from_array(
            np.array([array.item()], dtype=array.dtype), name
        )
        del tensor.dims[:]
        return tensor
    return numpy_helper.from_array(array, name)


def _ints_attr(node, name):
    attr = _attribute(node, name)
    if attr is None:
        return None
    return [int(value) for value in attr.ints]


def _vec(arrays, node, index):
    """A constant 1-D input, None if the input is omitted, or False if dynamic."""
    if index >= len(node.input) or node.input[index] == "":
        return None
    if node.input[index] not in arrays:
        return False
    return np.asarray(arrays[node.input[index]]).reshape(-1)


def _eval_unsqueeze(node, arrays):
    data = arrays.get(node.input[0]) if node.input else None
    if data is None:
        return None
    if len(node.input) > 1:
        axes = _vec(arrays, node, 1)
        if axes is False or axes is None:
            return None
        axes = [int(axis) for axis in axes]
    else:
        axes = _ints_attr(node, "axes")
        if axes is None:
            return None
    return np.expand_dims(data, axis=tuple(axes))


def _eval_squeeze(node, arrays):
    data = arrays.get(node.input[0]) if node.input else None
    if data is None:
        return None
    if len(node.input) > 1 and node.input[1] != "":
        axes = _vec(arrays, node, 1)
        if axes is False or axes is None:
            return None
        axes = tuple(int(axis) for axis in axes)
    else:
        axes = _ints_attr(node, "axes")
        axes = None if axes is None else tuple(axes)
    if not axes:
        return np.squeeze(data)
    return np.squeeze(data, axis=axes)


def _eval_concat(node, arrays):
    if not node.input or any(name not in arrays for name in node.input):
        return None
    axis_attr = _attribute(node, "axis")
    axis = 0 if axis_attr is None else int(axis_attr.i)
    return np.concatenate([arrays[name] for name in node.input], axis=axis)


def _eval_transpose(node, arrays):
    data = arrays.get(node.input[0]) if node.input else None
    if data is None:
        return None
    perm = _ints_attr(node, "perm")
    if perm is None:
        return np.transpose(data)
    return np.transpose(data, axes=perm)


def _eval_slice(node, arrays):
    data = arrays.get(node.input[0]) if node.input else None
    if data is None:
        return None
    # Opset >= 10 passes starts/ends/axes/steps as inputs. Older opsets use
    # attributes. A missing optional input falls back to the attribute.
    starts = _vec(arrays, node, 1)
    ends = _vec(arrays, node, 2)
    axes = _vec(arrays, node, 3)
    steps = _vec(arrays, node, 4)
    if starts is False or ends is False or axes is False or steps is False:
        return None
    if starts is None:
        starts = _ints_attr(node, "starts")
    if ends is None:
        ends = _ints_attr(node, "ends")
    if axes is None:
        axes = _ints_attr(node, "axes")
    if starts is None or ends is None:
        return None
    starts = [int(value) for value in np.asarray(starts).reshape(-1)]
    ends = [int(value) for value in np.asarray(ends).reshape(-1)]
    if axes is None:
        axes = list(range(len(starts)))
    else:
        axes = [int(value) for value in np.asarray(axes).reshape(-1)]
    if steps is None:
        steps = [1] * len(starts)
    else:
        steps = [int(value) for value in np.asarray(steps).reshape(-1)]
    slices = [slice(None)] * data.ndim
    for axis, start, end, step in zip(axes, starts, ends, steps):
        if axis < 0:
            axis += data.ndim
        slices[axis] = slice(start, end, step)
    return data[tuple(slices)]


_EVALUATORS = {
    "Unsqueeze": _eval_unsqueeze,
    "Squeeze": _eval_squeeze,
    "Concat": _eval_concat,
    "Transpose": _eval_transpose,
    "Slice": _eval_slice,
}


def fold_constants(graph, inherited=None):
    """Fold Constant and Identity-of-constant nodes into ``graph.initializer``.

    Parameters
    ----------
    graph: onnx.GraphProto
        Graph mutated in place. Folded nodes are removed unless a graph output
        is their only producer.
    inherited: dict
        Name to TensorProto constants already folded in an enclosing graph.

    Returns
    -------
    dict
        Name to TensorProto for every constant visible in this graph, including
        inherited ones.
    """
    weights = dict(inherited or {})
    arrays = {}
    for name, tensor in weights.items():
        try:
            array = numpy_helper.to_array(tensor)
        except Exception:
            continue
        if getattr(array.dtype, "kind", "") in ("O", "S", "U"):
            continue
        arrays[name] = array
    for tensor in graph.initializer:
        # A local initializer shadows a constant inherited from an outer graph.
        weights[tensor.name] = tensor
        try:
            array = numpy_helper.to_array(tensor)
        except Exception:
            arrays.pop(tensor.name, None)
            continue
        if getattr(array.dtype, "kind", "") in ("O", "S", "U"):
            arrays.pop(tensor.name, None)
            continue
        arrays[tensor.name] = array

    produced = {}
    changed = True
    while changed:
        changed = False
        for node in graph.node:
            if not node.output or node.output[0] in produced:
                continue
            # An initializer of the same name already supplies this value.
            if node.output[0] in weights and node.output[0] not in produced:
                if any(tensor.name == node.output[0] for tensor in graph.initializer):
                    continue
            array = None
            if node.op_type == "Constant":
                array = constant_node_to_array(node)
            elif (
                node.op_type == "Identity"
                and node.input
                and node.input[0] in arrays
                and len(node.output) == 1
            ):
                array = arrays[node.input[0]]
            elif node.op_type in _EVALUATORS:
                try:
                    array = _EVALUATORS[node.op_type](node, arrays)
                except Exception:
                    array = None
            if array is None or getattr(array.dtype, "kind", "") in ("O", "S", "U"):
                continue
            produced[node.output[0]] = _contiguous(array)
            arrays[node.output[0]] = produced[node.output[0]]
            changed = True

    outputs = {output.name for output in graph.output}
    kept = []
    for node in graph.node:
        folded = (
            node.output
            and node.output[0] in produced
            and node.op_type in ("Constant", "Identity", *_EVALUATORS)
        )
        if folded and not any(name in outputs for name in node.output):
            continue
        kept.append(node)
    del graph.node[:]
    graph.node.extend(kept)

    existing = {tensor.name for tensor in graph.initializer}
    for name, array in produced.items():
        if name in existing:
            continue
        tensor = _as_initializer(array, name)
        graph.initializer.append(tensor)
        weights[name] = tensor
    return weights
