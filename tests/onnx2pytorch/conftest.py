import os
import glob

import pytest
import onnx
import numpy as np
import onnxruntime as ort

from onnx2pytorch.utils import get_inputs_sample

RANDOM_SEED = 100
FIXTURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "fixtures"
)


@pytest.fixture(params=glob.glob(os.path.join(FIXTURES_DIR, "*.onnx")))
def onnx_model_path(request):
    return request.param


@pytest.fixture
def onnx_model(onnx_model_path):
    onnx_model = onnx.load(onnx_model_path)
    return onnx_model


@pytest.fixture
def onnx_inputs(onnx_model):
    np.random.seed(RANDOM_SEED)
    return get_inputs_sample(onnx_model)


@pytest.fixture
def onnx_model_outputs(onnx_model_path, onnx_model, onnx_inputs):
    ort_session = ort.InferenceSession(onnx_model_path)
    onnx_output = ort_session.run(None, onnx_inputs)
    return onnx_output


def pytest_terminal_summary(terminalreporter):
    """Report the opset matrix combinations that no runtime could adjudicate.

    These xfail rather than fail, so without this report they would be
    indistinguishable from passing cases. What the converter does for them is
    unverified, and the count is the size of that hole.
    """
    from tests.onnx2pytorch.opset_matrix import NO_ORACLE

    if not NO_ORACLE:
        return
    terminalreporter.write_sep(
        "=",
        "{} opset matrix combinations left unverified, no runtime could run "
        "them".format(len(NO_ORACLE)),
    )
    for op_type, opset, name, reason in sorted(NO_ORACLE):
        terminalreporter.write_line(
            "{}-{}-{}: {}".format(op_type, opset, name, reason.splitlines()[0][:160])
        )
