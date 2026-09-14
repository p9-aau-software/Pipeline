"""Metrics checked against values derived by hand.

If these drift, every number the pipeline has ever produced is suspect, so they are
written out the long way rather than by calling the implementation twice.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from recpipe.eval.metrics import ranking_metrics


def test_single_hit_and_single_miss():
    ranked = np.array([[1, 2, 3], [4, 5, 6]])
    relevant = [np.array([2]), np.array([7])]

    result = ranking_metrics(ranked, relevant, ks=[3], n_items=10)

    # User 0 hits at position 2 (1-indexed); user 1 misses entirely.
    assert result["recall@3"] == pytest.approx(0.5)
    assert result["precision@3"] == pytest.approx(1 / 6)
    assert result["hit_rate@3"] == pytest.approx(0.5)
    assert result["mrr@3"] == pytest.approx(0.25)

    expected_ndcg = (1 / math.log2(3)) / 1.0 / 2
    assert result["ndcg@3"] == pytest.approx(expected_ndcg)


def test_multiple_relevant_items():
    ranked = np.array([[0, 1, 2, 3]])
    relevant = [np.array([1, 3])]

    result = ranking_metrics(ranked, relevant, ks=[4], n_items=4)

    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)

    assert result["recall@4"] == pytest.approx(1.0)
    assert result["precision@4"] == pytest.approx(0.5)
    assert result["ndcg@4"] == pytest.approx(dcg / idcg)
    assert result["mrr@4"] == pytest.approx(0.5)


def test_perfect_ranking_scores_one():
    ranked = np.array([[5, 9], [1, 2]])
    relevant = [np.array([5]), np.array([1])]

    result = ranking_metrics(ranked, relevant, ks=[2], n_items=10)

    assert result["ndcg@2"] == pytest.approx(1.0)
    assert result["recall@2"] == pytest.approx(1.0)
    assert result["mrr@2"] == pytest.approx(1.0)


def test_coverage_counts_distinct_recommended_items():
    ranked = np.array([[0, 1], [0, 1]])
    relevant = [np.array([0]), np.array([1])]

    result = ranking_metrics(ranked, relevant, ks=[2], n_items=8)

    # Two distinct items recommended out of a catalogue of eight.
    assert result["coverage@2"] == pytest.approx(0.25)


def test_k_larger_than_ranking_is_clamped():
    ranked = np.array([[3, 4]])
    relevant = [np.array([4])]

    result = ranking_metrics(ranked, relevant, ks=[10])

    assert "ndcg@2" in result
    assert result["recall@2"] == pytest.approx(1.0)


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        ranking_metrics(np.array([[1, 2]]), [np.array([1]), np.array([2])], ks=[2])
