"""Differential test of every registered case at every schema revision."""

import pytest

from tests.onnx2pytorch.differential import (
    NoOracle,
    assert_outputs_match,
    make_single_node_model,
    min_ir_version,
    run_converted,
    run_oracle_strict,
)
from tests.onnx2pytorch.opset_matrix import CASES, NO_ORACLE, XFAILS, opsets_for


def _xfail_reason(op_type, opset, name):
    for key in ((op_type, opset, name), (op_type, None, name)):
        if key in XFAILS:
            return XFAILS[key]
    return None


def pytest_generate_tests(metafunc):
    if "matrix_case" not in metafunc.fixturenames:
        return
    params = []
    for op_type in sorted(CASES):
        for opset in opsets_for(op_type):
            for entry in CASES[op_type]:
                if not entry.applies_to(opset):
                    continue
                reason = _xfail_reason(op_type, opset, entry.name)
                marks = (
                    [pytest.mark.xfail(reason=reason, strict=True)] if reason else []
                )
                params.append(
                    pytest.param(
                        (op_type, opset, entry),
                        id="{}-{}-{}".format(op_type, opset, entry.name),
                        marks=marks,
                    )
                )
    metafunc.parametrize("matrix_case", params)


def test_matches_oracle(matrix_case):
    op_type, opset, entry = matrix_case
    model = make_single_node_model(
        op_type,
        entry.inputs,
        opset,
        outputs=entry.output_names,
        initializers=entry.initializers,
        input_names=entry.input_names,
        ir_version=min_ir_version(opset),
        **entry.attrs,
    )
    try:
        expected = run_oracle_strict(model, entry.inputs)
    except NoOracle as error:
        NO_ORACLE.append((op_type, opset, entry.name, str(error)))
        pytest.xfail("no oracle: {}".format(error))
    actual = run_converted(model, entry.inputs)
    assert_outputs_match(expected, actual, rtol=entry.rtol, atol=entry.atol)
