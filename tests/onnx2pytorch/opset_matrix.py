"""
Registry of single-node differential cases, crossed with ONNX schema revisions.

To cover a new operator, add an entry to ``CASES``::

    "Gemm": [
        case("default", {"a": rand(2, 3)}, initializers={"b": rand(3, 4)}),
        case("transB", {"a": rand(2, 3)}, initializers={"b": rand(4, 3)}, transB=1),
    ],

Every case is run at each opset where that operator's schema changed, which is
where cross-version regressions live. Restrict a case to the opsets where its
form is legal with ``since=``/``until=`` (for instance axes moving from an
attribute to an input). Node inputs that the converter needs at build time
(weights, shapes, split sizes) belong in ``initializers``; everything else is a
graph input. Known-unsupported combinations go in ``XFAILS``, keyed by
``(op_type, opset, case_name)`` with ``opset=None`` meaning every opset.
"""

from collections import defaultdict

import numpy as np
import onnx
from onnx import defs

MAX_OPSET = defs.onnx_opset_version()

# Combinations for which no runtime could produce a reference output.
NO_ORACLE = []

_rng = np.random.default_rng(0)


def rand(*shape, dtype=np.float32, scale=1.0):
    return (_rng.standard_normal(shape) * scale).astype(dtype)


def randint(*shape, low=0, high=8, dtype=np.int64):
    return _rng.integers(low, high, shape).astype(dtype)


def arr(values, dtype=np.float32):
    return np.array(values, dtype=dtype)


class Case:
    def __init__(
        self,
        name,
        inputs,
        attrs=None,
        initializers=None,
        input_names=None,
        num_outputs=1,
        since=1,
        until=MAX_OPSET,
        rtol=1e-5,
        atol=1e-6,
    ):
        self.name = name
        self.inputs = inputs
        self.attrs = attrs or {}
        self.initializers = initializers or {}
        self.input_names = input_names
        self.num_outputs = num_outputs
        self.since = since
        self.until = until
        self.rtol = rtol
        self.atol = atol

    def applies_to(self, opset):
        return self.since <= opset <= self.until

    @property
    def output_names(self):
        if self.num_outputs == 1:
            return ("y",)
        return tuple("y{}".format(i) for i in range(self.num_outputs))


def case(
    name,
    inputs,
    initializers=None,
    input_names=None,
    num_outputs=1,
    since=1,
    until=MAX_OPSET,
    rtol=1e-5,
    atol=1e-6,
    **attrs
):
    return Case(
        name,
        inputs,
        attrs=attrs,
        initializers=initializers,
        input_names=input_names,
        num_outputs=num_outputs,
        since=since,
        until=until,
        rtol=rtol,
        atol=atol,
    )


def _schema_revisions():
    revisions = defaultdict(set)
    for schema in defs.get_all_schemas_with_history():
        if schema.domain in ("", "ai.onnx") and schema.since_version <= MAX_OPSET:
            revisions[schema.name].add(schema.since_version)
    return revisions


SCHEMA_REVISIONS = _schema_revisions()


def opsets_for(op_type):
    """Opsets at which the operator's schema changed."""
    if op_type not in SCHEMA_REVISIONS:
        raise KeyError("No ai.onnx schema for {}".format(op_type))
    return sorted(SCHEMA_REVISIONS[op_type])


X4 = rand(2, 3, 5, 5)
X3 = rand(2, 3, 4)
X2 = rand(3, 4)
POS = np.abs(rand(2, 3, 4)) + 0.5
UNIT = rand(2, 3, 4) * 0.3

CASES = {}


def _elementwise(op_type, x=None, dtypes=("float64",), **attrs):
    x = X3 if x is None else x
    cases = [case("default", {"x": x}, **attrs)]
    for dtype in dtypes:
        cases.append(case(dtype, {"x": x.astype(dtype)}, **attrs))
    CASES[op_type] = cases


for _op in ("Abs", "Neg", "Ceil", "Floor", "Relu", "Sigmoid", "Tanh", "Erf", "Sign"):
    _elementwise(_op)
for _op in ("Exp", "Sinh", "Cosh", "Tan", "Atan", "Softplus", "Softsign", "Mish"):
    _elementwise(_op, UNIT)
for _op in ("Log", "Sqrt", "Reciprocal"):
    _elementwise(_op, POS)
for _op in ("Asin", "Acos", "Asinh", "Atanh"):
    _elementwise(_op, UNIT)
_elementwise("Round", rand(2, 3, 4, scale=3))
_elementwise("Acosh", POS + 1)
_elementwise("IsNaN", arr([0.0, np.nan, np.inf, -np.inf]), dtypes=())
_elementwise("IsInf", arr([0.0, np.nan, np.inf, -np.inf]), dtypes=())
_elementwise("Not", arr([True, False, True], dtype=bool), dtypes=())

CASES.update(
    {
        "Elu": [
            case("default", {"x": X3}),
            case("alpha", {"x": X3}, alpha=1.7),
        ],
        "LeakyRelu": [
            case("default", {"x": X3}),
            case("alpha", {"x": X3}, alpha=0.2),
        ],
        "Selu": [
            case("default", {"x": X3}),
            case("alpha_gamma", {"x": X3}, alpha=1.5, gamma=1.2),
        ],
        "Celu": [
            case("default", {"x": X3}),
            case("alpha", {"x": X3}, alpha=2.0),
        ],
        "ThresholdedRelu": [
            case("default", {"x": X3}),
            case("alpha", {"x": X3}, alpha=0.5),
        ],
        "HardSigmoid": [
            case("default", {"x": X3}),
            case("alpha_beta", {"x": X3}, alpha=0.1, beta=0.4),
        ],
        "HardSwish": [case("default", {"x": X3})],
        "Gelu": [
            case("default", {"x": X3}),
            case("tanh", {"x": X3}, approximate="tanh"),
        ],
        "PRelu": [
            case("scalar_slope", {"x": X3}, initializers={"slope": arr([0.25])}),
            case("per_channel", {"x": X3}, initializers={"slope": rand(3, 1)}),
        ],
        "Shrink": [
            case("default", {"x": X3}),
            case("bias_lambd", {"x": X3}, bias=0.5, lambd=0.7),
        ],
    }
)

# Binary elementwise, including the pre-7 broadcast/axis form.
for _op in ("Add", "Sub", "Mul", "Div"):
    CASES[_op] = [
        case("same_shape", {"a": X3, "b": np.abs(rand(2, 3, 4)) + 1}),
        case("numpy_broadcast", {"a": X3, "b": np.abs(rand(4)) + 1}, since=7),
        case(
            "legacy_broadcast_axis",
            {"a": X3, "b": np.abs(rand(3)) + 1},
            broadcast=1,
            axis=1,
            until=6,
        ),
        case(
            "legacy_broadcast_tail",
            {"a": X3, "b": np.abs(rand(4)) + 1},
            broadcast=1,
            until=6,
        ),
        case(
            "float64",
            {
                "a": X3.astype("float64"),
                "b": (np.abs(rand(2, 3, 4)) + 1).astype("float64"),
            },
        ),
        # Signed integers, where Div has to truncate towards zero rather than floor.
        case(
            "int64_signed",
            {
                "a": arr([[-7, -1, 0, 7], [-8, 8, -3, 3]], np.int64),
                "b": arr([[2, 2, 3, 2], [3, 3, -2, -2]], np.int64),
            },
            since=7,
        ),
    ]

CASES.update(
    {
        "Pow": [
            case("default", {"a": POS, "b": UNIT}),
            case("broadcast", {"a": POS, "b": arr([2.0])}, since=7),
        ],
        "Mod": [
            case(
                "int",
                {"a": randint(2, 3, low=1, high=20), "b": randint(2, 3, low=1, high=5)},
            ),
            case("fmod_float", {"a": X3, "b": np.abs(rand(2, 3, 4)) + 1}, fmod=1),
        ],
        "Min": [case("two", {"a": X3, "b": rand(2, 3, 4)})],
        "Max": [case("two", {"a": X3, "b": rand(2, 3, 4)})],
        "Sum": [
            case("two", {"a": X3, "b": rand(2, 3, 4)}),
            case("three", {"a": X3, "b": rand(2, 3, 4), "c": rand(2, 3, 4)}),
        ],
        "Mean": [case("three", {"a": X3, "b": rand(2, 3, 4), "c": rand(2, 3, 4)})],
        "Equal": [
            case("int", {"a": randint(2, 3, high=3), "b": randint(2, 3, high=3)})
        ],
        "Greater": [case("float", {"a": X3, "b": rand(2, 3, 4)}, since=7)],
        "GreaterOrEqual": [case("float", {"a": X3, "b": rand(2, 3, 4)}, since=12)],
        "Less": [case("float", {"a": X3, "b": rand(2, 3, 4)}, since=7)],
        "LessOrEqual": [case("float", {"a": X3, "b": rand(2, 3, 4)}, since=12)],
        "And": [
            case(
                "bool",
                {
                    "a": randint(2, 3, high=2).astype(bool),
                    "b": randint(2, 3, high=2).astype(bool),
                },
                since=7,
            )
        ],
        "Or": [
            case(
                "bool",
                {
                    "a": randint(2, 3, high=2).astype(bool),
                    "b": randint(2, 3, high=2).astype(bool),
                },
                since=7,
            )
        ],
        "Xor": [
            case(
                "bool",
                {
                    "a": randint(2, 3, high=2).astype(bool),
                    "b": randint(2, 3, high=2).astype(bool),
                },
                since=7,
            )
        ],
        "Where": [
            case(
                "default",
                {
                    "c": randint(2, 3, 4, high=2).astype(bool),
                    "a": X3,
                    "b": rand(2, 3, 4),
                },
            )
        ],
        "BitShift": [
            case(
                "left",
                {
                    "a": randint(2, 3, low=1, high=8, dtype=np.uint8),
                    "b": randint(2, 3, low=0, high=3, dtype=np.uint8),
                },
                direction="LEFT",
            ),
            case(
                "right",
                {
                    "a": randint(2, 3, low=1, high=200, dtype=np.uint8),
                    "b": randint(2, 3, low=0, high=3, dtype=np.uint8),
                },
                direction="RIGHT",
            ),
        ],
        "BitwiseAnd": [
            case(
                "int",
                {
                    "a": randint(2, 3, high=64, dtype=np.int32),
                    "b": randint(2, 3, high=64, dtype=np.int32),
                },
            )
        ],
        "BitwiseOr": [
            case(
                "int",
                {
                    "a": randint(2, 3, high=64, dtype=np.int32),
                    "b": randint(2, 3, high=64, dtype=np.int32),
                },
            )
        ],
        "BitwiseXor": [
            case(
                "int",
                {
                    "a": randint(2, 3, high=64, dtype=np.int32),
                    "b": randint(2, 3, high=64, dtype=np.int32),
                },
            )
        ],
        "BitwiseNot": [case("int", {"a": randint(2, 3, high=64, dtype=np.int32)})],
    }
)


# (op_type, opset, case_name) -> reason. opset=None applies to every opset.
# Marked strict, so fixing one of these without deleting the entry fails the suite.
XFAILS = {
    ("AveragePool", None, "dilations_same_upper"): (
        "no trustworthy oracle: with dilations onnxruntime derives the auto_pad "
        "pads from the undilated kernel, which contradicts its own shape "
        "inference, and onnx's reference puts all of the padding at the end"
    ),
    ("GRU", 1, "reverse"): (
        "no trustworthy oracle: onnxruntime has no GRU-1 kernel and onnx's "
        "reference GRU disagrees with onnxruntime for direction=reverse"
    ),
    ("GRU", 3, "reverse"): (
        "no trustworthy oracle: onnxruntime has no GRU-3 kernel and onnx's "
        "reference GRU disagrees with onnxruntime for direction=reverse"
    ),
    ("GRU", None, "clip"): "cell clipping has no torch equivalent",
    ("GRU", None, "initial_h"): "initial_h is rejected by convert_gru_layer",
    ("GRU", None, "layout"): "layout=1 is rejected by convert_gru_layer",
    ("LSTM", None, "clip"): "cell clipping has no torch equivalent",
    ("LSTM", None, "initial_h"): "initial_h is rejected by convert_lstm_layer",
    ("LSTM", None, "layout"): "layout=1 is rejected by convert_lstm_layer",
    ("LSTM", None, "peepholes"): "torch's LSTM has no peephole connections",
    (
        "LSTM",
        None,
        "input_forget",
    ): "torch's LSTM cannot couple the input and forget gates",
    ("RNN", None, "clip"): "cell clipping has no torch equivalent",
    ("RNN", None, "initial_h"): "initial_h is rejected by convert_rnn_layer",
    ("RNN", None, "layout"): "layout=1 is rejected by convert_rnn_layer",
    ("RNN", None, "sigmoid"): "torch's RNN only offers tanh and relu",
    ("LRN", None, "even_size"): (
        "no oracle: onnxruntime rejects even sizes and onnx's reference LRN sums "
        "over the batch axis instead of the channel axis"
    ),
}


IDX = arr([[0, 2], [1, 0]], dtype=np.int64)

CASES.update(
    {
        "Flatten": [
            case("default", {"x": rand(2, 3, 4, 5)}),
            case("axis0", {"x": rand(2, 3, 4, 5)}, axis=0),
            case("axis3", {"x": rand(2, 3, 4, 5)}, axis=3),
            case("negative_axis", {"x": rand(2, 3, 4, 5)}, axis=-2, since=11),
        ],
        "Reshape": [
            case(
                "initializer_shape",
                {"x": X3},
                initializers={"s": arr([4, 6], np.int64)},
                since=5,
            ),
            case(
                "minus_one",
                {"x": X3},
                initializers={"s": arr([2, -1], np.int64)},
                since=5,
            ),
            case(
                "zero_dim",
                {"x": X3},
                initializers={"s": arr([0, 12], np.int64)},
                since=5,
            ),
            case("dynamic_shape", {"x": X3, "s": arr([6, 4], np.int64)}, since=5),
            case("attribute_shape", {"x": X3}, shape=[4, 6], until=4),
        ],
        "Transpose": [
            case("default", {"x": rand(2, 3, 4)}),
            case("perm", {"x": rand(2, 3, 4)}, perm=[2, 0, 1]),
        ],
        "Concat": [
            case("axis0", {"a": X3, "b": rand(2, 3, 4)}, axis=0),
            case("axis2", {"a": X3, "b": rand(2, 3, 4)}, axis=2),
            case("negative_axis", {"a": X3, "b": rand(2, 3, 4)}, axis=-1, since=4),
        ],
        "Split": [
            case(
                "attribute_split",
                {"x": rand(2, 6)},
                axis=1,
                split=[2, 4],
                num_outputs=2,
                until=12,
            ),
            case("even_split", {"x": rand(2, 6)}, axis=1, num_outputs=2, until=12),
            case(
                "input_split",
                {"x": rand(2, 6)},
                initializers={"s": arr([2, 4], np.int64)},
                axis=1,
                num_outputs=2,
                since=13,
            ),
            case(
                "num_outputs_attr",
                {"x": rand(2, 6)},
                input_names=["x"],
                axis=1,
                num_outputs=3,
                since=18,
            ),
        ],
        "Slice": [
            case(
                "attributes",
                {"x": rand(4, 5)},
                starts=[1, 0],
                ends=[3, 4],
                axes=[0, 1],
                until=9,
            ),
            case(
                "inputs",
                {"x": rand(4, 5)},
                initializers={
                    "starts": arr([1, 0], np.int64),
                    "ends": arr([3, 4], np.int64),
                    "axes": arr([0, 1], np.int64),
                },
                since=10,
            ),
            case(
                "steps",
                {"x": rand(4, 5)},
                initializers={
                    "starts": arr([3, 4], np.int64),
                    "ends": arr([0, 0], np.int64),
                    "axes": arr([0, 1], np.int64),
                    "steps": arr([-1, -2], np.int64),
                },
                since=10,
            ),
        ],
        "Squeeze": [
            case("attribute_axes", {"x": rand(1, 3, 1, 4)}, axes=[0, 2], until=12),
            case("all_dims", {"x": rand(1, 3, 1, 4)}, until=12),
            case(
                "input_axes",
                {"x": rand(1, 3, 1, 4)},
                initializers={"a": arr([0, 2], np.int64)},
                since=13,
            ),
            case(
                "negative_axis",
                {"x": rand(1, 3, 1, 4)},
                initializers={"a": arr([-2], np.int64)},
                since=13,
            ),
        ],
        "Unsqueeze": [
            case("attribute_axes", {"x": rand(3, 4)}, axes=[0, 2], until=12),
            case(
                "input_axes",
                {"x": rand(3, 4)},
                initializers={"a": arr([0, 2], np.int64)},
                since=13,
            ),
        ],
        "Shape": [
            case("default", {"x": rand(2, 3, 4)}),
            case("start", {"x": rand(2, 3, 4)}, start=1, since=15),
            case("start_end", {"x": rand(2, 3, 4)}, start=0, end=2, since=15),
            case("negative_start", {"x": rand(2, 3, 4)}, start=-2, since=15),
        ],
        "Size": [case("default", {"x": rand(2, 3, 4)})],
        "Identity": [case("default", {"x": X3})],
        "Expand": [
            case(
                "broadcast",
                {"x": rand(3, 1)},
                initializers={"s": arr([2, 3, 4], np.int64)},
            ),
        ],
        "Tile": [
            case(
                "repeats",
                {"x": rand(2, 3)},
                initializers={"r": arr([2, 3], np.int64)},
                since=6,
            ),
        ],
        "Gather": [
            case("default", {"x": rand(5, 4), "i": IDX}),
            case("axis1", {"x": rand(5, 4), "i": IDX}, axis=1),
            case("negative_indices", {"x": rand(5, 4), "i": arr([[-1, -2]], np.int64)}),
        ],
        "GatherElements": [
            case(
                "axis0",
                {"x": rand(3, 3), "i": arr([[0, 1, 2], [2, 1, 0]], np.int64)},
                axis=0,
            ),
            case(
                "axis1",
                {"x": rand(3, 3), "i": arr([[0, 1, 2], [2, 1, 0]], np.int64)},
                axis=1,
            ),
        ],
        "GatherND": [
            case("default", {"x": rand(2, 3, 4), "i": arr([[0, 1], [1, 2]], np.int64)}),
            case(
                "batch_dims",
                {"x": rand(2, 3, 4), "i": arr([[[1]], [[0]]], np.int64)},
                batch_dims=1,
                since=12,
            ),
        ],
        "Scatter": [
            case(
                "axis1",
                {"x": rand(3, 4), "i": arr([[0, 2]], np.int64), "u": rand(1, 2)},
                axis=1,
            ),
        ],
        "ScatterElements": [
            case(
                "axis1",
                {"x": rand(3, 4), "i": arr([[0, 2]], np.int64), "u": rand(1, 2)},
                axis=1,
            ),
            case(
                "reduction_add",
                {"x": rand(3, 4), "i": arr([[0, 2]], np.int64), "u": rand(1, 2)},
                axis=1,
                reduction="add",
                since=16,
            ),
            case(
                "reduction_max",
                {"x": rand(3, 4), "i": arr([[0, 2]], np.int64), "u": rand(1, 2)},
                axis=1,
                reduction="max",
                since=18,
            ),
        ],
        "ScatterND": [
            case(
                "default",
                {"x": rand(4, 3), "i": arr([[0], [2]], np.int64), "u": rand(2, 3)},
            ),
            case(
                "reduction_add",
                {"x": rand(4, 3), "i": arr([[0], [2]], np.int64), "u": rand(2, 3)},
                reduction="add",
                since=16,
            ),
            case(
                "reduction_min",
                {"x": rand(4, 3), "i": arr([[0], [2]], np.int64), "u": rand(2, 3)},
                reduction="min",
                since=18,
            ),
        ],
        "Cast": [
            case(
                "float_to_int32",
                {"x": rand(2, 3, scale=8)},
                to=onnx.TensorProto.INT32,
                since=6,
            ),
            case(
                "int_to_float",
                {"x": randint(2, 3, high=9)},
                to=onnx.TensorProto.FLOAT,
                since=6,
            ),
            case(
                "float_to_double",
                {"x": rand(2, 3)},
                to=onnx.TensorProto.DOUBLE,
                since=6,
            ),
            case(
                "float_to_float16",
                {"x": rand(2, 3)},
                to=onnx.TensorProto.FLOAT16,
                since=6,
            ),
            case(
                "float_to_bool",
                {"x": arr([0.0, 1.0, -2.0])},
                to=onnx.TensorProto.BOOL,
                since=6,
            ),
        ],
        "CastLike": [
            case(
                "to_int32",
                {"x": rand(2, 3, scale=8), "t": randint(2, 3, dtype=np.int32)},
            ),
        ],
        "Clip": [
            case("attributes", {"x": rand(3, 4, scale=3)}, min=-1.0, max=1.0, until=10),
            case("min_only_attribute", {"x": rand(3, 4, scale=3)}, min=0.0, until=10),
            case(
                "inputs",
                {"x": rand(3, 4, scale=3)},
                initializers={"lo": arr(-1.0), "hi": arr(1.0)},
                since=11,
            ),
            case(
                "min_only_input",
                {"x": rand(3, 4, scale=3)},
                initializers={"lo": arr(0.0)},
                since=11,
            ),
        ],
        "ArgMax": [
            case("default", {"x": rand(3, 4)}),
            case("axis1", {"x": rand(3, 4)}, axis=1),
            case("keepdims0", {"x": rand(3, 4)}, axis=1, keepdims=0),
            case(
                "select_last_index",
                {"x": arr([[1.0, 3.0, 3.0]])},
                axis=1,
                select_last_index=1,
                since=12,
            ),
        ],
        "ArgMin": [
            case("default", {"x": rand(3, 4)}),
            case("axis1", {"x": rand(3, 4)}, axis=1),
            case("keepdims0", {"x": rand(3, 4)}, axis=1, keepdims=0),
        ],
        "TopK": [
            case("attribute_k", {"x": rand(3, 5)}, k=2, axis=1, num_outputs=2, until=9),
            case(
                "input_k",
                {"x": rand(3, 5)},
                initializers={"k": arr([2], np.int64)},
                axis=1,
                num_outputs=2,
                since=10,
            ),
            case(
                "smallest",
                {"x": rand(3, 5)},
                initializers={"k": arr([2], np.int64)},
                axis=1,
                largest=0,
                num_outputs=2,
                since=11,
            ),
            case(
                "axis0",
                {"x": rand(3, 5)},
                initializers={"k": arr([2], np.int64)},
                axis=0,
                num_outputs=2,
                since=10,
            ),
        ],
        "DepthToSpace": [
            case("dcr", {"x": rand(1, 8, 2, 3)}, blocksize=2),
            case("crd", {"x": rand(1, 8, 2, 3)}, blocksize=2, mode="CRD", since=11),
        ],
        "SpaceToDepth": [case("default", {"x": rand(1, 2, 4, 6)}, blocksize=2)],
        "OneHot": [
            case(
                "default",
                {"i": arr([1, 3], np.int64)},
                initializers={"d": arr([5], np.int64), "v": arr([0.0, 1.0])},
            ),
            case(
                "axis0",
                {"i": arr([1, 3], np.int64)},
                initializers={"d": arr([5], np.int64), "v": arr([0.0, 1.0])},
                axis=0,
            ),
        ],
        "NonZero": [case("default", {"x": arr([[0.0, 1.0], [2.0, 0.0]])})],
        "CumSum": [
            case("default", {"x": rand(2, 4)}, initializers={"a": arr(1, np.int64)}),
            case(
                "reverse",
                {"x": rand(2, 4)},
                initializers={"a": arr(1, np.int64)},
                reverse=1,
            ),
            case(
                "exclusive",
                {"x": rand(2, 4)},
                initializers={"a": arr(1, np.int64)},
                exclusive=1,
            ),
        ],
        "EyeLike": [
            case("default", {"x": rand(3, 4)}),
            case("k", {"x": rand(3, 4)}, k=1),
        ],
        "Trilu": [
            case("upper", {"x": rand(4, 4)}),
            case("lower", {"x": rand(4, 4)}, upper=0),
            case("k", {"x": rand(4, 4)}, initializers={"k": arr(1, np.int64)}),
        ],
        "Compress": [
            case(
                "axis0", {"x": rand(3, 4), "c": arr([True, False, True], bool)}, axis=0
            ),
            case(
                "flat",
                {
                    "x": rand(3, 4),
                    "c": arr(
                        [
                            True,
                            False,
                            True,
                            False,
                            True,
                            False,
                            True,
                            False,
                            True,
                            False,
                            True,
                            False,
                        ],
                        bool,
                    ),
                },
            ),
        ],
        "Einsum": [
            case("matmul", {"a": rand(2, 3), "b": rand(3, 4)}, equation="ij,jk->ik"),
            case("transpose", {"a": rand(2, 3)}, equation="ij->ji"),
        ],
        "Range": [
            case(
                "int",
                {
                    "start": arr(0, np.int64),
                    "limit": arr(10, np.int64),
                    "delta": arr(3, np.int64),
                },
            ),
        ],
        "ConstantOfShape": [
            case("default", {"s": arr([2, 3], np.int64)}),
        ],
        "ReverseSequence": [
            case(
                "default",
                {"x": rand(4, 3), "lens": arr([4, 2, 3], np.int64)},
                time_axis=0,
                batch_axis=1,
            ),
        ],
    }
)

# Softmax family: the pre-13 versions coerce the input to 2D around the axis.
for _op in ("Softmax", "LogSoftmax", "Hardmax"):
    CASES[_op] = [
        case("default", {"x": rand(2, 3, 4)}),
        case("axis0", {"x": rand(2, 3, 4)}, axis=0),
        case("axis2", {"x": rand(2, 3, 4)}, axis=2),
        case("negative_axis", {"x": rand(2, 3, 4)}, axis=-1, since=11),
    ]

_REDUCTIONS = (
    "ReduceMean",
    "ReduceMax",
    "ReduceMin",
    "ReduceProd",
    "ReduceSum",
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceSumSquare",
)

for _op in _REDUCTIONS:
    _x = POS if _op in ("ReduceLogSum", "ReduceProd") else rand(2, 3, 4)
    # Axes moved from an attribute to an input at opset 13 for ReduceSum, 18 elsewhere.
    _input_axes_since = 13 if _op == "ReduceSum" else 18
    CASES[_op] = [
        case("all_axes", {"x": _x}, until=_input_axes_since - 1),
        case("axis1", {"x": _x}, axes=[1], until=_input_axes_since - 1),
        case("two_axes", {"x": _x}, axes=[0, 2], until=_input_axes_since - 1),
        case("keepdims0", {"x": _x}, axes=[1], keepdims=0, until=_input_axes_since - 1),
        case("negative_axis", {"x": _x}, axes=[-1], until=_input_axes_since - 1),
        case("all_axes_input", {"x": _x}, input_names=["x"], since=_input_axes_since),
        case(
            "axis1_input",
            {"x": _x},
            initializers={"a": arr([1], np.int64)},
            since=_input_axes_since,
        ),
        case(
            "two_axes_input",
            {"x": _x},
            initializers={"a": arr([0, 2], np.int64)},
            since=_input_axes_since,
        ),
        case(
            "keepdims0_input",
            {"x": _x},
            initializers={"a": arr([1], np.int64)},
            keepdims=0,
            since=_input_axes_since,
        ),
        case(
            "noop_with_empty_axes",
            {"x": _x},
            input_names=["x"],
            noop_with_empty_axes=1,
            since=_input_axes_since,
        ),
    ]


CONV_X = rand(1, 2, 6, 6)
CONV_W = rand(3, 2, 3, 3)
CONV_B = rand(3)

CASES.update(
    {
        "Conv": [
            case(
                "default",
                {"x": CONV_X},
                initializers={"w": CONV_W},
                kernel_shape=[3, 3],
            ),
            case("no_kernel_shape", {"x": CONV_X}, initializers={"w": CONV_W}),
            case(
                "bias",
                {"x": CONV_X},
                initializers={"w": CONV_W, "b": CONV_B},
                kernel_shape=[3, 3],
            ),
            case(
                "pads",
                {"x": CONV_X},
                initializers={"w": CONV_W},
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
            ),
            case(
                "asymmetric_pads",
                {"x": CONV_X},
                initializers={"w": CONV_W},
                kernel_shape=[3, 3],
                pads=[0, 1, 2, 1],
            ),
            case(
                "strides",
                {"x": CONV_X},
                initializers={"w": CONV_W},
                kernel_shape=[3, 3],
                strides=[2, 2],
            ),
            case(
                "dilations",
                {"x": CONV_X},
                initializers={"w": CONV_W},
                kernel_shape=[3, 3],
                dilations=[2, 2],
            ),
            case(
                "grouped",
                {"x": rand(1, 4, 6, 6)},
                initializers={"w": rand(4, 2, 3, 3)},
                kernel_shape=[3, 3],
                group=2,
            ),
            case(
                "depthwise",
                {"x": rand(1, 4, 6, 6)},
                initializers={"w": rand(4, 1, 3, 3)},
                kernel_shape=[3, 3],
                group=4,
            ),
            case(
                "same_upper",
                {"x": CONV_X},
                initializers={"w": CONV_W},
                kernel_shape=[3, 3],
                auto_pad="SAME_UPPER",
            ),
            case(
                "same_lower",
                {"x": CONV_X},
                initializers={"w": CONV_W},
                kernel_shape=[3, 3],
                auto_pad="SAME_LOWER",
            ),
            case(
                "same_upper_stride2",
                {"x": CONV_X},
                initializers={"w": CONV_W},
                kernel_shape=[3, 3],
                strides=[2, 2],
                auto_pad="SAME_UPPER",
            ),
            case(
                "same_lower_stride2",
                {"x": CONV_X},
                initializers={"w": CONV_W},
                kernel_shape=[3, 3],
                strides=[2, 2],
                auto_pad="SAME_LOWER",
            ),
            case(
                "same_upper_even_kernel",
                {"x": CONV_X},
                initializers={"w": rand(3, 2, 2, 2)},
                kernel_shape=[2, 2],
                auto_pad="SAME_UPPER",
            ),
            case(
                "same_lower_even_kernel",
                {"x": CONV_X},
                initializers={"w": rand(3, 2, 2, 2)},
                kernel_shape=[2, 2],
                auto_pad="SAME_LOWER",
            ),
            case(
                "valid",
                {"x": CONV_X},
                initializers={"w": CONV_W},
                kernel_shape=[3, 3],
                auto_pad="VALID",
            ),
            case(
                "conv1d",
                {"x": rand(1, 2, 8)},
                initializers={"w": rand(3, 2, 3)},
                kernel_shape=[3],
                pads=[1, 1],
            ),
            case(
                "conv1d_same_upper",
                {"x": rand(1, 2, 8)},
                initializers={"w": rand(3, 2, 3)},
                kernel_shape=[3],
                auto_pad="SAME_UPPER",
            ),
            case(
                "conv3d",
                {"x": rand(1, 2, 4, 4, 4)},
                initializers={"w": rand(3, 2, 2, 2, 2)},
                kernel_shape=[2, 2, 2],
            ),
            case(
                "conv3d_pads",
                {"x": rand(1, 2, 4, 4, 4)},
                initializers={"w": rand(3, 2, 3, 3, 3)},
                kernel_shape=[3, 3, 3],
                pads=[1, 1, 1, 1, 1, 1],
            ),
            case(
                "float64",
                {"x": CONV_X.astype("float64")},
                initializers={"w": CONV_W.astype("float64")},
                kernel_shape=[3, 3],
            ),
        ],
        "ConvTranspose": [
            case(
                "default",
                {"x": rand(1, 2, 5, 5)},
                initializers={"w": rand(2, 3, 3, 3)},
                kernel_shape=[3, 3],
            ),
            case(
                "bias",
                {"x": rand(1, 2, 5, 5)},
                initializers={"w": rand(2, 3, 3, 3), "b": rand(3)},
                kernel_shape=[3, 3],
            ),
            case(
                "pads",
                {"x": rand(1, 2, 5, 5)},
                initializers={"w": rand(2, 3, 3, 3)},
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
            ),
            case(
                "strides",
                {"x": rand(1, 2, 5, 5)},
                initializers={"w": rand(2, 3, 3, 3)},
                kernel_shape=[3, 3],
                strides=[2, 2],
            ),
            case(
                "dilations",
                {"x": rand(1, 2, 5, 5)},
                initializers={"w": rand(2, 3, 3, 3)},
                kernel_shape=[3, 3],
                dilations=[2, 2],
            ),
            case(
                "grouped",
                {"x": rand(1, 4, 5, 5)},
                initializers={"w": rand(4, 2, 3, 3)},
                kernel_shape=[3, 3],
                group=2,
            ),
            case(
                "output_padding",
                {"x": rand(1, 2, 5, 5)},
                initializers={"w": rand(2, 3, 3, 3)},
                kernel_shape=[3, 3],
                strides=[2, 2],
                output_padding=[1, 1],
            ),
            case(
                "output_shape",
                {"x": rand(1, 2, 5, 5)},
                initializers={"w": rand(2, 3, 3, 3)},
                kernel_shape=[3, 3],
                strides=[2, 2],
                output_shape=[10, 10],
            ),
            case(
                "output_shape_odd",
                {"x": rand(1, 2, 5, 5)},
                initializers={"w": rand(2, 3, 3, 3)},
                kernel_shape=[3, 3],
                strides=[2, 2],
                output_shape=[9, 9],
            ),
            case(
                "output_shape_same_upper",
                {"x": rand(1, 2, 5, 5)},
                initializers={"w": rand(2, 3, 3, 3)},
                kernel_shape=[3, 3],
                strides=[2, 2],
                output_shape=[9, 9],
                auto_pad="SAME_UPPER",
            ),
            case(
                "asymmetric_pads",
                {"x": rand(1, 2, 5, 5)},
                initializers={"w": rand(2, 3, 3, 3)},
                kernel_shape=[3, 3],
                pads=[0, 1, 2, 1],
            ),
            case(
                "same_upper",
                {"x": rand(1, 2, 5, 5)},
                initializers={"w": rand(2, 3, 3, 3)},
                kernel_shape=[3, 3],
                strides=[2, 2],
                auto_pad="SAME_UPPER",
            ),
            case(
                "same_lower",
                {"x": rand(1, 2, 5, 5)},
                initializers={"w": rand(2, 3, 3, 3)},
                kernel_shape=[3, 3],
                strides=[2, 2],
                auto_pad="SAME_LOWER",
            ),
            case(
                "same_upper_dilations",
                {"x": rand(1, 2, 5, 5)},
                initializers={"w": rand(2, 3, 3, 3)},
                kernel_shape=[3, 3],
                strides=[2, 2],
                dilations=[2, 2],
                auto_pad="SAME_UPPER",
            ),
            case(
                "valid",
                {"x": rand(1, 2, 5, 5)},
                initializers={"w": rand(2, 3, 3, 3)},
                kernel_shape=[3, 3],
                auto_pad="VALID",
            ),
            case(
                "convtranspose1d",
                {"x": rand(1, 2, 6)},
                initializers={"w": rand(2, 3, 3)},
                kernel_shape=[3],
            ),
        ],
        "AveragePool": [
            case("default", {"x": CONV_X}, kernel_shape=[3, 3]),
            case("strides", {"x": CONV_X}, kernel_shape=[3, 3], strides=[2, 2]),
            case("pads", {"x": CONV_X}, kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            case(
                "count_include_pad",
                {"x": CONV_X},
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
                count_include_pad=1,
                since=7,
            ),
            case(
                "ceil_mode",
                {"x": CONV_X},
                kernel_shape=[3, 3],
                strides=[2, 2],
                ceil_mode=1,
                since=10,
            ),
            # Before opset 7 the padded cells count towards the average anyway
            case(
                "same_upper_legacy",
                {"x": CONV_X},
                kernel_shape=[3, 3],
                auto_pad="SAME_UPPER",
                until=6,
            ),
            case(
                "same_upper",
                {"x": CONV_X},
                kernel_shape=[3, 3],
                auto_pad="SAME_UPPER",
                since=7,
            ),
            case(
                "same_lower",
                {"x": CONV_X},
                kernel_shape=[3, 3],
                auto_pad="SAME_LOWER",
                since=7,
            ),
            case(
                "same_upper_count_include_pad",
                {"x": CONV_X},
                kernel_shape=[3, 3],
                auto_pad="SAME_UPPER",
                count_include_pad=1,
                since=7,
            ),
            case(
                "asymmetric_pads",
                {"x": CONV_X},
                kernel_shape=[3, 3],
                pads=[1, 0, 2, 1],
                since=7,
            ),
            case(
                "asymmetric_pads_count_include_pad",
                {"x": CONV_X},
                kernel_shape=[3, 3],
                pads=[1, 0, 2, 1],
                count_include_pad=1,
                since=7,
            ),
            case("valid", {"x": CONV_X}, kernel_shape=[3, 3], auto_pad="VALID"),
            case("avgpool1d", {"x": rand(1, 2, 8)}, kernel_shape=[3], strides=[2]),
            case("avgpool3d", {"x": rand(1, 2, 4, 4, 4)}, kernel_shape=[2, 2, 2]),
            case(
                "dilations",
                {"x": CONV_X},
                kernel_shape=[2, 2],
                dilations=[2, 2],
                since=19,
            ),
            case(
                "dilations_strides",
                {"x": CONV_X},
                kernel_shape=[2, 2],
                dilations=[2, 2],
                strides=[2, 2],
                since=19,
            ),
            case(
                "dilations_pads",
                {"x": CONV_X},
                kernel_shape=[2, 2],
                dilations=[2, 2],
                pads=[1, 1, 1, 1],
                count_include_pad=1,
                since=19,
            ),
            case(
                "dilations_pads_exclude",
                {"x": CONV_X},
                kernel_shape=[2, 2],
                dilations=[2, 2],
                pads=[1, 1, 1, 1],
                since=19,
            ),
            case(
                "dilations_same_upper",
                {"x": CONV_X},
                kernel_shape=[2, 2],
                dilations=[2, 2],
                auto_pad="SAME_UPPER",
                since=19,
            ),
            case(
                "dilations1d",
                {"x": rand(1, 2, 8)},
                kernel_shape=[3],
                dilations=[2],
                since=19,
            ),
        ],
        "MaxPool": [
            case("default", {"x": CONV_X}, kernel_shape=[3, 3]),
            case("strides", {"x": CONV_X}, kernel_shape=[3, 3], strides=[2, 2]),
            case("pads", {"x": CONV_X}, kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            case(
                "indices",
                {"x": CONV_X},
                kernel_shape=[2, 2],
                strides=[2, 2],
                num_outputs=2,
                since=8,
            ),
            case(
                "storage_order",
                {"x": CONV_X},
                kernel_shape=[2, 2],
                strides=[2, 2],
                storage_order=1,
                num_outputs=2,
                since=8,
            ),
            case(
                "ceil_mode",
                {"x": CONV_X},
                kernel_shape=[3, 3],
                strides=[2, 2],
                ceil_mode=1,
                since=10,
            ),
            case(
                "dilations",
                {"x": CONV_X},
                kernel_shape=[2, 2],
                dilations=[2, 2],
                since=10,
            ),
            case(
                "same_upper", {"x": CONV_X}, kernel_shape=[3, 3], auto_pad="SAME_UPPER"
            ),
            case(
                "same_lower", {"x": CONV_X}, kernel_shape=[3, 3], auto_pad="SAME_LOWER"
            ),
            case("valid", {"x": CONV_X}, kernel_shape=[3, 3], auto_pad="VALID"),
            case("maxpool1d", {"x": rand(1, 2, 8)}, kernel_shape=[3], strides=[2]),
            case("maxpool3d", {"x": rand(1, 2, 4, 4, 4)}, kernel_shape=[2, 2, 2]),
            case("float64", {"x": CONV_X.astype("float64")}, kernel_shape=[3, 3]),
        ],
        "GlobalAveragePool": [case("default", {"x": CONV_X})],
        "GlobalMaxPool": [case("default", {"x": CONV_X})],
        "GlobalLpPool": [
            case("default", {"x": np.abs(CONV_X) + 0.5}),
            case("p1", {"x": np.abs(CONV_X) + 0.5}, p=1, since=2),
        ],
        "LpPool": [
            case("default", {"x": np.abs(CONV_X) + 0.5}, kernel_shape=[3, 3], since=2),
            case("p1", {"x": np.abs(CONV_X) + 0.5}, kernel_shape=[3, 3], p=1, since=2),
            case(
                "strides",
                {"x": np.abs(CONV_X) + 0.5},
                kernel_shape=[2, 2],
                strides=[2, 2],
                since=2,
            ),
        ],
        "BatchNormalization": [
            case(
                "default",
                {"x": rand(2, 3, 4, 4)},
                initializers={
                    "scale": np.abs(rand(3)) + 0.5,
                    "b": rand(3),
                    "mean": rand(3),
                    "var": np.abs(rand(3)) + 0.5,
                },
                since=7,
            ),
            case(
                "epsilon",
                {"x": rand(2, 3, 4, 4)},
                initializers={
                    "scale": np.abs(rand(3)) + 0.5,
                    "b": rand(3),
                    "mean": rand(3),
                    "var": np.abs(rand(3)) + 0.5,
                },
                epsilon=1e-2,
                since=7,
            ),
            case(
                "rank3",
                {"x": rand(2, 3, 5)},
                initializers={
                    "scale": np.abs(rand(3)) + 0.5,
                    "b": rand(3),
                    "mean": rand(3),
                    "var": np.abs(rand(3)) + 0.5,
                },
                since=7,
            ),
        ],
        "InstanceNormalization": [
            case(
                "default",
                {"x": rand(2, 3, 4, 4)},
                initializers={"scale": np.abs(rand(3)) + 0.5, "b": rand(3)},
                since=6,
            ),
            case(
                "epsilon",
                {"x": rand(2, 3, 4, 4)},
                initializers={"scale": np.abs(rand(3)) + 0.5, "b": rand(3)},
                epsilon=1e-2,
                since=6,
            ),
        ],
        "LayerNormalization": [
            case(
                "default",
                {"x": rand(2, 3, 4)},
                initializers={"scale": np.abs(rand(4)) + 0.5},
            ),
            case(
                "bias",
                {"x": rand(2, 3, 4)},
                initializers={"scale": np.abs(rand(4)) + 0.5, "b": rand(4)},
            ),
            case(
                "axis1",
                {"x": rand(2, 3, 4)},
                initializers={"scale": np.abs(rand(3, 4)) + 0.5},
                axis=1,
            ),
        ],
        "GroupNormalization": [
            case(
                "default",
                {"x": rand(2, 4, 3, 3)},
                initializers={"scale": np.abs(rand(2)) + 0.5, "b": rand(2)},
                num_groups=2,
                until=20,
            ),
            case(
                "per_channel",
                {"x": rand(2, 4, 3, 3)},
                initializers={"scale": np.abs(rand(4)) + 0.5, "b": rand(4)},
                num_groups=2,
                since=21,
            ),
        ],
        "LRN": [
            case("default", {"x": rand(2, 5, 3, 3)}, size=3),
            case("even_size", {"x": rand(2, 5, 3, 3)}, size=4),
            case(
                "alpha_beta_bias",
                {"x": rand(2, 5, 3, 3)},
                size=3,
                alpha=1e-3,
                beta=0.6,
                bias=1.5,
            ),
        ],
        "MeanVarianceNormalization": [
            case("default", {"x": rand(2, 3, 4, 4)}),
            case("axes", {"x": rand(2, 3, 4, 4)}, axes=[0, 1]),
        ],
        "LpNormalization": [
            case("default", {"x": rand(2, 3, 4)}),
            case("p1_axis0", {"x": rand(2, 3, 4)}, p=1, axis=0),
        ],
        "Gemm": [
            case("default", {"a": rand(2, 3)}, initializers={"b": rand(3, 4)}, since=7),
            case(
                "bias",
                {"a": rand(2, 3)},
                initializers={"b": rand(3, 4), "c": rand(4)},
                since=7,
            ),
            case(
                "alpha_beta",
                {"a": rand(2, 3)},
                initializers={"b": rand(3, 4), "c": rand(4)},
                alpha=0.5,
                beta=2.0,
                since=7,
            ),
            case(
                "transB",
                {"a": rand(2, 3)},
                initializers={"b": rand(4, 3)},
                transB=1,
                since=7,
            ),
            case(
                "transA",
                {"a": rand(3, 2)},
                initializers={"b": rand(3, 4)},
                transA=1,
                since=7,
            ),
            case(
                "legacy_broadcast",
                {"a": rand(2, 3)},
                initializers={"b": rand(3, 4), "c": rand(4)},
                broadcast=1,
                until=6,
            ),
        ],
        "MatMul": [
            case(
                "initializer_weight", {"a": rand(2, 3)}, initializers={"b": rand(3, 4)}
            ),
            case("dynamic_2d", {"a": rand(2, 3), "b": rand(3, 4)}),
            case("batched", {"a": rand(2, 3, 4), "b": rand(2, 4, 5)}),
            case("broadcast_batch", {"a": rand(2, 3, 4), "b": rand(4, 5)}, since=9),
        ],
    }
)


SEQ, BATCH, INPUT, HIDDEN = 3, 2, 4, 5
RNN_X = rand(SEQ, BATCH, INPUT, scale=0.5)

# A batch holding every interesting length: full, partial and empty.
VAR_X = rand(SEQ, 4, INPUT, scale=0.5)
VAR_LENS = arr([SEQ, 1, 0, SEQ - 1], np.int32)


def _rnn_weights(gates, directions=1):
    return {
        "w": rand(directions, gates * HIDDEN, INPUT, scale=0.3),
        "r": rand(directions, gates * HIDDEN, HIDDEN, scale=0.3),
        "b": rand(directions, 2 * gates * HIDDEN, scale=0.3),
    }


def _rnn_cases(gates, num_outputs, extra=()):
    weights = _rnn_weights(gates)
    bidir = _rnn_weights(gates, 2)
    cases = [
        case(
            "forward",
            {"x": RNN_X},
            initializers=weights,
            hidden_size=HIDDEN,
            num_outputs=num_outputs,
        ),
        case(
            "no_bias",
            {"x": RNN_X},
            initializers={"w": weights["w"], "r": weights["r"]},
            hidden_size=HIDDEN,
            num_outputs=num_outputs,
        ),
        case(
            "bidirectional",
            {"x": RNN_X},
            initializers=bidir,
            hidden_size=HIDDEN,
            direction="bidirectional",
            num_outputs=num_outputs,
        ),
        case(
            "reverse",
            {"x": RNN_X},
            initializers=weights,
            hidden_size=HIDDEN,
            direction="reverse",
            num_outputs=num_outputs,
        ),
        case(
            "clip",
            {"x": RNN_X},
            initializers=weights,
            hidden_size=HIDDEN,
            clip=0.5,
            num_outputs=num_outputs,
        ),
        case(
            "initial_h",
            {"x": RNN_X},
            initializers=dict(weights, h0=rand(1, BATCH, HIDDEN, scale=0.3)),
            input_names=["x", "w", "r", "b", "", "h0"],
            hidden_size=HIDDEN,
            num_outputs=num_outputs,
        ),
        case(
            "sequence_lens",
            {"x": RNN_X},
            initializers=dict(weights, seq_lens=arr([SEQ, SEQ - 1], np.int32)),
            hidden_size=HIDDEN,
            num_outputs=num_outputs,
        ),
        case(
            "sequence_lens_mixed",
            {"x": VAR_X},
            initializers=dict(weights, seq_lens=VAR_LENS),
            hidden_size=HIDDEN,
            num_outputs=num_outputs,
        ),
        case(
            "sequence_lens_reverse",
            {"x": VAR_X},
            initializers=dict(weights, seq_lens=VAR_LENS),
            hidden_size=HIDDEN,
            direction="reverse",
            num_outputs=num_outputs,
        ),
        case(
            "sequence_lens_bidirectional",
            {"x": VAR_X},
            initializers=dict(bidir, seq_lens=VAR_LENS),
            hidden_size=HIDDEN,
            direction="bidirectional",
            num_outputs=num_outputs,
        ),
        case(
            "layout",
            {"x": rand(BATCH, SEQ, INPUT, scale=0.5)},
            initializers=weights,
            hidden_size=HIDDEN,
            layout=1,
            num_outputs=num_outputs,
            since=14,
        ),
    ]
    cases.extend(extra)
    return cases


CASES.update(
    {
        "RNN": _rnn_cases(
            1,
            2,
            extra=[
                case(
                    "relu",
                    {"x": RNN_X},
                    initializers=_rnn_weights(1),
                    hidden_size=HIDDEN,
                    activations=["Relu"],
                    num_outputs=2,
                ),
                case(
                    "sigmoid",
                    {"x": RNN_X},
                    initializers=_rnn_weights(1),
                    hidden_size=HIDDEN,
                    activations=["Sigmoid"],
                    num_outputs=2,
                ),
            ],
        ),
        "GRU": _rnn_cases(
            3,
            2,
            extra=[
                case(
                    "linear_before_reset",
                    {"x": RNN_X},
                    initializers=_rnn_weights(3),
                    hidden_size=HIDDEN,
                    linear_before_reset=1,
                    num_outputs=2,
                    since=3,
                ),
                case(
                    "sequence_lens_linear_before_reset",
                    {"x": VAR_X},
                    initializers=dict(_rnn_weights(3), seq_lens=VAR_LENS),
                    hidden_size=HIDDEN,
                    linear_before_reset=1,
                    num_outputs=2,
                    since=3,
                ),
            ],
        ),
        "LSTM": _rnn_cases(
            4,
            3,
            extra=[
                case(
                    "peepholes",
                    {"x": RNN_X},
                    initializers=dict(
                        _rnn_weights(4), p=rand(1, 3 * HIDDEN, scale=0.3)
                    ),
                    input_names=["x", "w", "r", "b", "", "", "", "p"],
                    hidden_size=HIDDEN,
                    num_outputs=3,
                ),
                case(
                    "input_forget",
                    {"x": RNN_X},
                    initializers=_rnn_weights(4),
                    hidden_size=HIDDEN,
                    input_forget=1,
                    num_outputs=3,
                ),
            ],
        ),
    }
)


PAD_X = rand(2, 3)
Q_X = rand(2, 4, scale=2)
Q_SCALE = arr(0.05)
Q_ZERO = np.array(128, dtype=np.uint8)

CASES.update(
    {
        "Pad": [
            case("legacy_paddings", {"x": PAD_X}, paddings=[1, 2, 1, 2], until=1),
            case("attribute_pads", {"x": PAD_X}, pads=[1, 2, 1, 2], since=2, until=10),
            case(
                "attribute_value",
                {"x": PAD_X},
                pads=[1, 2, 1, 2],
                value=3.5,
                since=2,
                until=10,
            ),
            case(
                "attribute_reflect",
                {"x": PAD_X},
                pads=[1, 1, 1, 1],
                mode="reflect",
                since=2,
                until=10,
            ),
            case(
                "attribute_edge",
                {"x": PAD_X},
                pads=[1, 1, 1, 1],
                mode="edge",
                since=2,
                until=10,
            ),
            case(
                "input_pads",
                {"x": PAD_X},
                initializers={"p": arr([1, 2, 1, 2], np.int64)},
                since=11,
            ),
            case(
                "input_value",
                {"x": PAD_X},
                initializers={"p": arr([1, 2, 1, 2], np.int64), "v": arr(3.5)},
                since=11,
            ),
            case(
                "input_reflect",
                {"x": PAD_X},
                initializers={"p": arr([1, 1, 1, 1], np.int64)},
                mode="reflect",
                since=11,
            ),
            case(
                "input_edge",
                {"x": PAD_X},
                initializers={"p": arr([1, 1, 1, 1], np.int64)},
                mode="edge",
                since=11,
            ),
            case(
                "negative_pads",
                {"x": PAD_X},
                initializers={"p": arr([-1, 0, 0, -1], np.int64)},
                since=11,
            ),
            case(
                "axes",
                {"x": PAD_X},
                initializers={
                    "p": arr([1, 2], np.int64),
                    "v": arr(0.0),
                    "a": arr([1], np.int64),
                },
                since=18,
            ),
            case(
                "wrap",
                {"x": PAD_X},
                initializers={"p": arr([1, 1, 1, 1], np.int64)},
                mode="wrap",
                since=19,
            ),
        ],
        "Resize": [
            case(
                "nearest_scales",
                {"x": rand(1, 1, 2, 2)},
                initializers={"s": arr([1.0, 1.0, 2.0, 2.0])},
                until=10,
            ),
            case(
                "nearest_scales_roi",
                {"x": rand(1, 1, 2, 2)},
                initializers={"s": arr([1.0, 1.0, 2.0, 2.0])},
                input_names=["x", "", "s"],
                since=11,
            ),
            case(
                "linear_align_corners",
                {"x": rand(1, 1, 2, 2)},
                initializers={"s": arr([1.0, 1.0, 2.0, 2.0])},
                input_names=["x", "", "s"],
                mode="linear",
                coordinate_transformation_mode="align_corners",
                since=11,
            ),
            case(
                "sizes",
                {"x": rand(1, 1, 2, 2)},
                initializers={"z": arr([1, 1, 4, 4], np.int64)},
                input_names=["x", "", "", "z"],
                mode="linear",
                coordinate_transformation_mode="align_corners",
                since=11,
            ),
        ],
        "Upsample": [
            case(
                "legacy_scale_attributes",
                {"x": rand(1, 1, 2, 2)},
                height_scale=2.0,
                width_scale=2.0,
                mode="nearest",
                until=6,
            ),
            case(
                "attribute_scales",
                {"x": rand(1, 1, 2, 2)},
                scales=[1.0, 1.0, 2.0, 2.0],
                since=7,
                until=8,
            ),
            case(
                "input_scales",
                {"x": rand(1, 1, 2, 2)},
                initializers={"s": arr([1.0, 1.0, 2.0, 2.0])},
                since=9,
            ),
            case(
                "input_scales_linear",
                {"x": rand(1, 1, 2, 2)},
                initializers={"s": arr([1.0, 1.0, 2.0, 2.0])},
                mode="linear",
                since=9,
            ),
        ],
        "Dropout": [
            case("inference", {"x": X3}, since=7),
            case("ratio_attribute", {"x": X3}, ratio=0.5, since=7, until=11),
            case("mask", {"x": X3}, num_outputs=2, since=7),
            case("ratio_input", {"x": X3}, initializers={"r": arr(0.5)}, since=12),
            case(
                "mask_input_ratio",
                {"x": X3},
                initializers={"r": arr(0.5)},
                num_outputs=2,
                since=12,
            ),
        ],
        "QuantizeLinear": [
            case("scalar_scale", {"x": Q_X}, initializers={"s": Q_SCALE, "z": Q_ZERO}),
            case("no_zero_point", {"x": Q_X}, initializers={"s": Q_SCALE}),
            case(
                "per_axis",
                {"x": Q_X},
                initializers={
                    "s": np.full(4, 0.05, np.float32),
                    "z": np.full(4, 128, np.uint8),
                },
                axis=1,
                since=13,
            ),
        ],
        "DequantizeLinear": [
            case(
                "scalar_scale",
                {"x": randint(2, 4, high=255, dtype=np.uint8)},
                initializers={"s": Q_SCALE, "z": Q_ZERO},
            ),
            case(
                "per_axis",
                {"x": randint(2, 4, high=255, dtype=np.uint8)},
                initializers={
                    "s": np.full(4, 0.05, np.float32),
                    "z": np.full(4, 128, np.uint8),
                },
                axis=1,
                since=13,
            ),
        ],
        "DynamicQuantizeLinear": [case("default", {"x": Q_X}, num_outputs=3)],
        "MatMulInteger": [
            case(
                "zero_points",
                {
                    "a": randint(2, 3, high=255, dtype=np.uint8),
                    "b": randint(3, 4, high=255, dtype=np.uint8),
                },
                initializers={
                    "az": np.array(120, np.uint8),
                    "bz": np.array(130, np.uint8),
                },
            ),
            case(
                "no_zero_points",
                {
                    "a": randint(2, 3, high=255, dtype=np.uint8),
                    "b": randint(3, 4, high=255, dtype=np.uint8),
                },
            ),
        ],
        "ConvInteger": [
            case(
                "default",
                {"x": randint(1, 1, 5, 5, high=255, dtype=np.uint8)},
                initializers={"w": randint(2, 1, 3, 3, high=255, dtype=np.uint8)},
            ),
            case(
                "zero_points",
                {"x": randint(1, 1, 5, 5, high=255, dtype=np.uint8)},
                initializers={
                    "w": randint(2, 1, 3, 3, high=255, dtype=np.uint8),
                    "xz": np.array(120, np.uint8),
                    "wz": np.array(130, np.uint8),
                },
            ),
        ],
        "QLinearMatMul": [
            case(
                "default",
                {"a": randint(2, 3, high=255, dtype=np.uint8)},
                initializers={
                    "as": Q_SCALE,
                    "az": Q_ZERO,
                    "b": randint(3, 4, high=255, dtype=np.uint8),
                    "bs": Q_SCALE,
                    "bz": Q_ZERO,
                    "ys": arr(0.1),
                    "yz": Q_ZERO,
                },
            )
        ],
        "QLinearConv": [
            case(
                "default",
                {"x": randint(1, 1, 5, 5, high=255, dtype=np.uint8)},
                initializers={
                    "xs": Q_SCALE,
                    "xz": Q_ZERO,
                    "w": randint(2, 1, 3, 3, high=255, dtype=np.uint8),
                    "ws": Q_SCALE,
                    "wz": Q_ZERO,
                    "ys": arr(0.5),
                    "yz": Q_ZERO,
                },
            )
        ],
        "SoftmaxCrossEntropyLoss": [
            case("mean", {"x": rand(3, 5), "t": randint(3, high=5)}),
            case("none", {"x": rand(3, 5), "t": randint(3, high=5)}, reduction="none"),
            case("sum", {"x": rand(3, 5), "t": randint(3, high=5)}, reduction="sum"),
        ],
        "NegativeLogLikelihoodLoss": [
            case("mean", {"x": rand(3, 5), "t": randint(3, high=5)}),
            case("none", {"x": rand(3, 5), "t": randint(3, high=5)}, reduction="none"),
        ],
        "Det": [case("default", {"x": rand(2, 3, 3)})],
        "Unique": [
            case("sorted", {"x": arr([2.0, 1.0, 1.0, 3.0])}, num_outputs=4),
            case(
                "axis",
                {"x": arr([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]])},
                axis=0,
                num_outputs=4,
            ),
        ],
        "GridSample": [
            case(
                "bilinear",
                {
                    "x": rand(1, 1, 4, 4),
                    "g": (rand(1, 3, 3, 2) * 0.5).astype("float32"),
                },
            ),
            case(
                "nearest",
                {
                    "x": rand(1, 1, 4, 4),
                    "g": (rand(1, 3, 3, 2) * 0.5).astype("float32"),
                },
                mode="nearest",
                since=20,
            ),
        ],
        "CenterCropPad": [
            case("crop", {"x": rand(4, 6)}, initializers={"s": arr([2, 3], np.int64)}),
            case("pad", {"x": rand(4, 6)}, initializers={"s": arr([6, 8], np.int64)}),
        ],
        "Col2Im": [
            # x is (N, C * prod(block_shape), number of blocks)
            case(
                "default",
                {"x": rand(1, 4, 9)},
                initializers={"s": arr([4, 4], np.int64), "b": arr([2, 2], np.int64)},
            ),
            case(
                "channels",
                {"x": rand(2, 12, 9)},
                initializers={"s": arr([4, 4], np.int64), "b": arr([2, 2], np.int64)},
            ),
            case(
                "strides_dilations",
                {"x": rand(1, 4, 4)},
                initializers={"s": arr([6, 6], np.int64), "b": arr([2, 2], np.int64)},
                strides=[2, 2],
                dilations=[2, 2],
            ),
            case(
                "pads",
                {"x": rand(1, 4, 25)},
                initializers={"s": arr([4, 4], np.int64), "b": arr([2, 2], np.int64)},
                pads=[1, 1, 1, 1],
            ),
            case(
                "one_dimensional",
                {"x": rand(1, 2, 5)},
                initializers={"s": arr([6], np.int64), "b": arr([2], np.int64)},
            ),
            case(
                "three_dimensional",
                {"x": rand(1, 8, 24)},
                initializers={
                    "s": arr([3, 4, 5], np.int64),
                    "b": arr([2, 2, 2], np.int64),
                },
            ),
        ],
        "MaxRoiPool": [
            case(
                "default",
                {"x": rand(1, 2, 6, 6), "rois": arr([[0, 0, 0, 4, 4]])},
                pooled_shape=[2, 2],
            )
        ],
        "RoiAlign": [
            case(
                "default",
                {
                    "x": rand(1, 2, 6, 6),
                    "rois": arr([[0.0, 0.0, 4.0, 4.0]]),
                    "bi": arr([0], np.int64),
                },
                output_height=2,
                output_width=2,
            )
        ],
    }
)


CASES.update(
    {
        "Constant": [
            case(
                "value",
                {},
                value=onnx.helper.make_tensor(
                    "v", onnx.TensorProto.FLOAT, [2, 2], [1.0, 2.0, 3.0, 4.0]
                ),
            ),
            case("value_floats", {}, value_floats=[1.0, 2.0], since=12),
            case("value_ints", {}, value_ints=[1, 2], since=12),
        ],
        "IsInf": [
            case("default", {"x": arr([1.0, np.inf, -np.inf, np.nan])}),
            case(
                "no_negative",
                {"x": arr([1.0, np.inf, -np.inf, np.nan])},
                detect_negative=0,
            ),
            case(
                "no_positive",
                {"x": arr([1.0, np.inf, -np.inf, np.nan])},
                detect_positive=0,
            ),
        ],
        "MaxUnpool": [
            case(
                "default",
                {"x": rand(1, 1, 2, 2), "i": arr([[[[0, 3], [12, 15]]]], np.int64)},
                kernel_shape=[2, 2],
                strides=[2, 2],
            )
        ],
        "CumProd": [
            case(
                "default",
                {"x": np.abs(rand(2, 4)) + 0.5},
                initializers={"a": arr(1, np.int64)},
            ),
            case(
                "reverse",
                {"x": np.abs(rand(2, 4)) + 0.5},
                initializers={"a": arr(1, np.int64)},
                reverse=1,
            ),
        ],
        "Swish": [
            case("default", {"x": X3}),
            case("alpha", {"x": X3}, alpha=1.5),
        ],
        "RMSNormalization": [
            case(
                "default",
                {"x": rand(2, 3, 4)},
                initializers={"scale": np.abs(rand(4)) + 0.5},
            ),
        ],
        "HannWindow": [case("default", {"n": arr(8, np.int64)})],
        "HammingWindow": [case("default", {"n": arr(8, np.int64)})],
        "BlackmanWindow": [case("default", {"n": arr(8, np.int64)})],
        "AffineGrid": [
            case(
                "default",
                {"theta": arr([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])},
                initializers={"s": arr([1, 1, 3, 3], np.int64)},
            )
        ],
    }
)

# A handful of non-float32 paths, which the operator tests barely touch.
CASES["Softmax"].append(case("float64", {"x": rand(2, 3, 4).astype("float64")}))
CASES["Gemm"].append(
    case(
        "float64",
        {"a": rand(2, 3).astype("float64")},
        initializers={"b": rand(3, 4).astype("float64")},
        since=7,
    )
)
CASES["MatMul"].append(
    case(
        "int32",
        {"a": randint(2, 3, dtype=np.int32), "b": randint(3, 4, dtype=np.int32)},
    )
)
CASES["Transpose"].append(case("int64", {"x": randint(2, 3, 4)}, perm=[2, 0, 1]))
CASES["Concat"].append(
    case("int64_axis0", {"a": randint(2, 3), "b": randint(2, 3)}, axis=0)
)
CASES["Gather"].append(
    case("int32_data", {"x": randint(5, 4, dtype=np.int32), "i": IDX})
)
