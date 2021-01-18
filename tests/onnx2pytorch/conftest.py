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
