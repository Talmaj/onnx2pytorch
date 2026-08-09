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
XFAILS = {}
