"""Keras GRU exported by tf2onnx, the graph behind onnx2torch issue 233.

Generated with TensorFlow 2.20 and tf2onnx 1.17 at opset 14::

    inp = tf.keras.Input(shape=(5, 4), batch_size=2, name="input")
    out = tf.keras.layers.GRU(3, return_sequences=True, name="gru")(inp)
    model = tf.keras.Model(inp, out)
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=(tf.TensorSpec((2, 5, 4), tf.float32, name="input"),),
        opset=14,
    )

Current tf2onnx lowers that layer to a Loop of Gemm/Sigmoid/Tanh rather than
one GRU node. The Gemm weights are loop inputs, which is the same
initializer-only failure as a dynamic linear layer. The explicit-activations
GRU cases live in test_recurrent_attributes.py.
"""

import os

import numpy as np
import onnx
import onnxruntime as ort
import torch

from onnx2pytorch.convert import ConvertModel

FIXTURE = os.path.join(os.path.dirname(__file__), "data", "keras_gru_opset14.onnx")


def test_keras_gru_matches_onnxruntime():
    model = onnx.load(FIXTURE)
    rng = np.random.default_rng(0)
    feeds = {"input": rng.standard_normal((2, 5, 4)).astype(np.float32)}
    expected = ort.InferenceSession(FIXTURE, providers=["CPUExecutionProvider"]).run(
        None, feeds
    )[0]
    converted = ConvertModel(model)
    converted.eval()
    with torch.no_grad():
        actual = converted(torch.from_numpy(feeds["input"])).detach().cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
