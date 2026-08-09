import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel

# Pool holding the unigrams 2, 3, 5, 4 followed by the bigrams (5, 6), (7, 8), (6, 7)
POOL = [2, 3, 5, 4, 5, 6, 7, 8, 6, 7]
NGRAM_COUNTS = [0, 4]
NGRAM_INDEXES = [0, 1, 2, 3, 4, 5, 6]


def build_model(x, **attrs):
    node = helper.make_node("TfIdfVectorizer", inputs=["x"], outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "tfidfvectorizer_test",
        [helper.make_tensor_value_info("x", TensorProto.INT32, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 9)])


def check_tfidf_vectorizer(x, **attrs):
    attrs.setdefault("ngram_counts", NGRAM_COUNTS)
    attrs.setdefault("ngram_indexes", NGRAM_INDEXES)
    attrs.setdefault("pool_int64s", POOL)
    model = build_model(x, **attrs)

    exp_y = ort.InferenceSession(model.SerializeToString()).run(None, {"x": x})[0]
    with torch.no_grad():
        y = ConvertModel(model)(torch.from_numpy(x))
    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-6, atol=1e-6)


def test_tfidf_vectorizer_unigrams():
    x = np.array([1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8], dtype=np.int32)
    check_tfidf_vectorizer(
        x, mode="TF", min_gram_length=1, max_gram_length=1, max_skip_count=0
    )


def test_tfidf_vectorizer_bigrams():
    x = np.array([1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8], dtype=np.int32)
    check_tfidf_vectorizer(
        x, mode="TF", min_gram_length=2, max_gram_length=2, max_skip_count=0
    )


def test_tfidf_vectorizer_uni_and_bigrams():
    x = np.array([1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8], dtype=np.int32)
    check_tfidf_vectorizer(
        x, mode="TF", min_gram_length=1, max_gram_length=2, max_skip_count=0
    )


@pytest.mark.parametrize("max_skip_count", [0, 1, 2, 5])
def test_tfidf_vectorizer_skip_count(max_skip_count):
    x = np.array([1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8], dtype=np.int32)
    check_tfidf_vectorizer(
        x,
        mode="TF",
        min_gram_length=2,
        max_gram_length=2,
        max_skip_count=max_skip_count,
    )


def test_tfidf_vectorizer_batch():
    x = np.array(
        [[1, 1, 3, 3, 3, 7], [8, 6, 7, 5, 6, 8]],
        dtype=np.int32,
    )
    check_tfidf_vectorizer(
        x, mode="TF", min_gram_length=1, max_gram_length=2, max_skip_count=0
    )


@pytest.mark.parametrize("mode", ["TF", "IDF", "TFIDF"])
def test_tfidf_vectorizer_modes(mode):
    x = np.array([1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8], dtype=np.int32)
    check_tfidf_vectorizer(
        x, mode=mode, min_gram_length=1, max_gram_length=2, max_skip_count=0
    )


@pytest.mark.parametrize("mode", ["IDF", "TFIDF"])
def test_tfidf_vectorizer_weights(mode):
    x = np.array([1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8], dtype=np.int32)
    check_tfidf_vectorizer(
        x,
        mode=mode,
        min_gram_length=1,
        max_gram_length=2,
        max_skip_count=0,
        weights=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
    )


def test_tfidf_vectorizer_no_match():
    x = np.array([100, 101, 102], dtype=np.int32)
    check_tfidf_vectorizer(
        x, mode="TF", min_gram_length=1, max_gram_length=2, max_skip_count=0
    )


def test_tfidf_vectorizer_pool_strings_not_implemented():
    x = np.array([1, 2], dtype=np.int32)
    model = build_model(
        x,
        mode="TF",
        min_gram_length=1,
        max_gram_length=1,
        max_skip_count=0,
        ngram_counts=[0],
        ngram_indexes=[0, 1],
        pool_strings=["a", "b"],
    )
    with pytest.raises(NotImplementedError):
        ConvertModel(model)
