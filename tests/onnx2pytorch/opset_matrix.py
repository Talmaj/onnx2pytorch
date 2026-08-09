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
        "Less": [case("float", {"a": X3, "b": rand(2, 3, 4)}, since=7)],
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
    ("AveragePool", None, "same_upper"): (
        "auto_pad prepends the padding as input values, so count_include_pad=0 "
        "cannot exclude it again"
    ),
    ("AveragePool", None, "dilations"): "torch's AvgPool has no dilation argument",
    ("ConvTranspose", None, "output_shape"): (
        "output_shape needs the pads to be derived and cropped off the output"
    ),
    ("ConvTranspose", None, "same_upper"): (
        "auto_pad pads the input, ConvTranspose instead has to crop the output"
    ),
    ("GRU", None, "reverse"): "the converter only builds forward or bidirectional",
    ("GRU", None, "clip"): "cell clipping has no torch equivalent",
    ("GRU", None, "initial_h"): "initial_h is rejected by convert_gru_layer",
    ("GRU", None, "sequence_lens"): (
        "sequence_lens is parsed but never applied, so padded steps are not zeroed"
    ),
    ("GRU", None, "layout"): "layout=1 is rejected by convert_gru_layer",
    ("LSTM", None, "reverse"): "the converter only builds forward or bidirectional",
    ("LSTM", None, "clip"): "cell clipping has no torch equivalent",
    ("LSTM", None, "initial_h"): "initial_h is rejected by convert_lstm_layer",
    ("LSTM", None, "sequence_lens"): (
        "sequence_lens is parsed but never applied, so padded steps are not zeroed"
    ),
    ("LSTM", None, "layout"): "layout=1 is rejected by convert_lstm_layer",
    ("LSTM", None, "peepholes"): "torch's LSTM has no peephole connections",
    (
        "LSTM",
        None,
        "input_forget",
    ): "torch's LSTM cannot couple the input and forget gates",
    ("RNN", None, "reverse"): "the converter only builds forward or bidirectional",
    ("RNN", None, "clip"): "cell clipping has no torch equivalent",
    ("RNN", None, "initial_h"): "initial_h is rejected by convert_rnn_layer",
    ("RNN", None, "sequence_lens"): (
        "sequence_lens is parsed but never applied, so padded steps are not zeroed"
    ),
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
                "same_upper",
                {"x": rand(1, 2, 5, 5)},
                initializers={"w": rand(2, 3, 3, 3)},
                kernel_shape=[3, 3],
                strides=[2, 2],
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
                "same_upper_count_include_pad",
                {"x": CONV_X},
                kernel_shape=[3, 3],
                auto_pad="SAME_UPPER",
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
