import numpy as np
import pytest

from engine import (
    top_k_indices,
    mmr_rerank,
    compute_scores,
    build_matrices,
    scale_score,
    format_score,
    stability_analysis,
)


class TestTopKIndices:
    def test_returns_k_indices(self):
        scores = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        idx = top_k_indices(scores, 3)
        assert len(idx) == 3

    def test_sorted_best_first(self):
        scores = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        idx = top_k_indices(scores, 5)
        for i in range(len(idx) - 1):
            assert scores[idx[i]] >= scores[idx[i + 1]]

    def test_deterministic_on_ties(self):
        scores = np.array([0.5, 0.5, 0.5, 0.4, 0.4])
        idx1 = top_k_indices(scores, 5)
        idx2 = top_k_indices(scores, 5)
        assert list(idx1) == list(idx2)

    def test_k_larger_than_length(self):
        scores = np.array([0.1, 0.5])
        idx = top_k_indices(scores, 10)
        assert len(idx) == 2


class TestMMRRerank:
    def test_returns_unique_indices(self):
        scores = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.0])
        n = len(scores)
        pool = np.arange(n)
        embeddings = np.eye(n, dtype=np.float32)
        result = mmr_rerank(pool, scores, embeddings, 0.8, 5)
        assert len(result) == len(set(result))

    def test_starts_with_top_scored(self):
        scores = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
        n = len(scores)
        pool = np.arange(n)
        embeddings = np.eye(n, dtype=np.float32)
        result = mmr_rerank(pool, scores, embeddings, 0.8, 3)
        assert result[0] == 0  # index 0 has highest score

    def test_lambda_1_pure_score(self):
        scores = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
        n = len(scores)
        pool = np.arange(n)
        embeddings = np.eye(n, dtype=np.float32)
        result = mmr_rerank(pool, scores, embeddings, 1.0, 5)
        sorted_scores = sorted(range(len(scores)), key=lambda i: -scores[i])
        assert result == sorted_scores

    def test_empty_pool_returns_empty(self):
        assert mmr_rerank(np.array([]), np.array([]), np.empty((0, 4)), 0.8, 5) == []


class TestComputeScores:
    def test_basic_computation(self):
        n = 3
        subscore_matrix = np.ones((n, 4), dtype=np.float32)
        penalties = np.ones(n, dtype=np.float32)
        semantic_sim = np.ones(n, dtype=np.float32)
        weights = {"technical_fit": 0.25, "career_quality": 0.25,
                    "availability_signal": 0.25, "seniority_fit": 0.15,
                    "semantic_similarity": 0.10}
        scores = compute_scores(subscore_matrix, penalties, semantic_sim, weights)
        assert scores.shape == (n,)
        assert np.allclose(scores, [1.0, 1.0, 1.0])

    def test_penalty_applied(self):
        n = 2
        subscore_matrix = np.ones((n, 4), dtype=np.float32)
        penalties = np.array([1.0, 0.5], dtype=np.float32)
        semantic_sim = np.ones(n, dtype=np.float32)
        weights = {"technical_fit": 0.25, "career_quality": 0.25,
                    "availability_signal": 0.25, "seniority_fit": 0.15,
                    "semantic_similarity": 0.10}
        scores = compute_scores(subscore_matrix, penalties, semantic_sim, weights)
        assert scores[0] > scores[1]


class TestBuildMatrices:
    def test_build_from_dicts(self):
        ids = np.array(["a", "b"])
        subscores = {
            "a": {"technical_fit": 0.8, "career_quality": 0.6,
                  "availability_signal": 0.4, "seniority_fit": 0.2,
                  "penalty_multiplier": 1.0},
            "b": {"technical_fit": 0.7, "career_quality": 0.5,
                  "availability_signal": 0.3, "seniority_fit": 0.1,
                  "penalty_multiplier": 0.5},
        }
        sm, pen = build_matrices(ids, subscores)
        assert sm.shape == (2, 4)
        assert list(pen) == [1.0, 0.5]
        assert sm[0, 0] == 0.8


class TestScaleScore:
    def test_default_scale(self):
        assert scale_score(0.5) == 5.0

    def test_clamping(self):
        assert scale_score(-0.1) == 0.0
        assert scale_score(1.5) == 10.0

    def test_custom_scale(self):
        assert scale_score(0.5, 100) == 50.0


class TestFormatScore:
    def test_formatting(self):
        result = format_score(0.856, scale=10, decimals=2)
        assert result == "8.56"

    def test_format_with_tie_precision(self):
        a = format_score(0.8567, scale=10, decimals=3)
        b = format_score(0.8562, scale=10, decimals=3)
        assert a == "8.567"
        assert b == "8.562"
        assert a != b


class TestStabilityAnalysis:
    def test_returns_dict(self):
        n = 10
        subscore_matrix = np.random.randn(n, 4).astype(np.float32)
        penalties = np.ones(n, dtype=np.float32)
        semantic_sim = np.random.randn(n).astype(np.float32)
        weights = {"technical_fit": 0.25, "career_quality": 0.25,
                    "availability_signal": 0.25, "seniority_fit": 0.15,
                    "semantic_similarity": 0.10}
        result = stability_analysis(subscore_matrix, penalties, semantic_sim,
                                     weights, k=3, n_trials=10, seed=42)
        assert isinstance(result, dict)
