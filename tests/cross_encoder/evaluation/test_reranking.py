"""Tests for CrossEncoderRerankingEvaluator batched prediction."""

from __future__ import annotations

import math
from unittest.mock import Mock

import pytest

from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator


def test_negative_format(reranker_bert_tiny_model: CrossEncoder) -> None:
    """Basic test with 'negative' format produces valid metrics."""
    samples = [
        {
            "query": "What is Python?",
            "positive": ["Python is a programming language"],
            "negative": ["Java is a language", "C++ is fast"],
        },
        {
            "query": "What is the sun?",
            "positive": ["The sun is a star"],
            "negative": ["The moon orbits earth", "Mars is red"],
        },
        {"query": "What is water?", "positive": ["Water is H2O"], "negative": ["Salt is NaCl", "Gold is Au"]},
    ]
    evaluator = CrossEncoderRerankingEvaluator(samples, name="neg-test")
    results = evaluator(reranker_bert_tiny_model)

    assert "neg-test_map" in results
    assert f"neg-test_mrr@{evaluator.at_k}" in results
    assert f"neg-test_ndcg@{evaluator.at_k}" in results
    for value in results.values():
        assert 0.0 <= value <= 1.0


def test_documents_format(reranker_bert_tiny_model: CrossEncoder) -> None:
    """Test with 'documents' format produces both base and reranked metrics."""
    samples = [
        {
            "query": "What is Python?",
            "positive": ["Python is a programming language"],
            "documents": ["Java is a language", "Python is a programming language", "C++ is fast"],
        },
        {
            "query": "What is the sun?",
            "positive": ["The sun is a star"],
            "documents": ["The moon orbits earth", "The sun is a star", "Mars is red"],
        },
    ]
    evaluator = CrossEncoderRerankingEvaluator(samples, name="doc-test")
    results = evaluator(reranker_bert_tiny_model)

    # Should have both base and reranked metrics
    assert "doc-test_map" in results
    assert "doc-test_base_map" in results
    assert f"doc-test_ndcg@{evaluator.at_k}" in results
    assert f"doc-test_base_ndcg@{evaluator.at_k}" in results


def test_no_relevant_docs_in_documents(reranker_bert_tiny_model: CrossEncoder) -> None:
    """Edge case: positive not in documents list."""
    samples = [
        {"query": "empty", "positive": ["x"], "documents": ["a", "b", "c"]},
    ]
    evaluator = CrossEncoderRerankingEvaluator(samples, name="edge-test")
    results = evaluator(reranker_bert_tiny_model)

    assert f"edge-test_ndcg@{evaluator.at_k}" in results


def test_single_sample(reranker_bert_tiny_model: CrossEncoder) -> None:
    """Single sample still works."""
    samples = [
        {"query": "test", "positive": ["yes"], "negative": ["no"]},
    ]
    evaluator = CrossEncoderRerankingEvaluator(samples, name="single")
    results = evaluator(reranker_bert_tiny_model)

    assert f"single_ndcg@{evaluator.at_k}" in results


def test_string_positive(reranker_bert_tiny_model: CrossEncoder) -> None:
    """Positive as a string (not list) is handled correctly."""
    samples = [
        {"query": "test", "positive": "yes", "negative": ["no", "nope"]},
    ]
    evaluator = CrossEncoderRerankingEvaluator(samples, name="str-pos")
    results = evaluator(reranker_bert_tiny_model)

    assert f"str-pos_ndcg@{evaluator.at_k}" in results


def test_validation_errors() -> None:
    """Invalid samples raise clear errors."""
    with pytest.raises(ValueError, match="query"):
        CrossEncoderRerankingEvaluator([{"positive": ["a"], "negative": ["b"]}])(None)

    with pytest.raises(ValueError, match="positive"):
        CrossEncoderRerankingEvaluator([{"query": "q", "negative": ["b"]}])(None)

    with pytest.raises(ValueError, match="exactly one"):
        CrossEncoderRerankingEvaluator([{"query": "q", "positive": ["a"]}])(None)

    with pytest.raises(ValueError, match="exactly one"):
        CrossEncoderRerankingEvaluator([{"query": "q", "positive": ["a"], "negative": ["b"], "documents": ["c"]}])(
            None
        )


def test_empty_documents() -> None:
    samples = [
        {"query": "empty", "positive": ["missing"], "documents": []},
        {"query": "found", "positive": ["positive"], "documents": ["positive", "negative"]},
    ]
    model = Mock()
    model.predict.return_value = [1.0, 0.0]

    results = CrossEncoderRerankingEvaluator(samples, always_rerank_positives=False)(model)

    assert results == {
        "map": 0.5,
        "mrr@10": 0.5,
        "ndcg@10": 0.5,
        "base_map": 0.5,
        "base_mrr@10": 0.5,
        "base_ndcg@10": 0.5,
    }
    model.predict.assert_called_once()
    assert model.predict.call_args.args[0] == [["found", "positive"], ["found", "negative"]]


def test_single_document_samples() -> None:
    """A sample with exactly one candidate document is trivially ranked first and must not crash.

    Both the base metric pass (documents format) and the reranked metric pass used to raise
    "Computing NDCG is only meaningful when there is more than 1 document" from sklearn.
    """
    samples = [
        # documents format: the first-stage retrieval only returned the gold document
        {"query": "q1", "positive": ["gold"], "documents": ["gold"]},
        # negative format: the only candidate is the positive itself
        {"query": "q2", "positive": ["gold"], "negative": []},
    ]
    model = Mock()
    model.predict.return_value = [0.9, 0.8]

    results = CrossEncoderRerankingEvaluator(samples, always_rerank_positives=False)(model)

    # One relevant document ranks perfectly: MRR, NDCG@10 and AP are all 1.0 for both samples
    assert results == {
        "map": 1.0,
        "mrr@10": 1.0,
        "ndcg@10": 1.0,
        "base_map": 1.0,
        "base_mrr@10": 1.0,
        "base_ndcg@10": 1.0,
    }


def test_empty_samples() -> None:
    """An evaluator without any samples logs and returns instead of crashing on the dataset stats."""
    model = Mock()

    results = CrossEncoderRerankingEvaluator([], write_csv=False)(model)

    model.predict.assert_not_called()
    assert all(math.isnan(value) for value in results.values())
