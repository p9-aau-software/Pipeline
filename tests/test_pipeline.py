"""End-to-end on synthetic data: ingest, prepare, split, fit, evaluate.

Runs on CPU in seconds. This is the check to run before spending a GPU slot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recpipe import models as _models  # noqa: F401  (registers the models)
from recpipe.data.dataset import InteractionMatrix
from recpipe.data.prepare import prepare
from recpipe.data.splits import build_splits, leaked_pairs
from recpipe.eval.evaluator import evaluate_model, evaluate_predictions
from recpipe.models.base import TrainContext
from recpipe.registry import MODELS

DATA_CFG = {
    "name": "synthetic",
    "ingest": "synthetic",
    "params": {"n_users": 200, "n_items": 80, "interactions_per_user": 25, "seed": 1},
    "filter": {"deduplicate": True, "min_user_interactions": 5, "min_item_interactions": 5},
}


@pytest.fixture
def splits(tmp_path):
    prepared = prepare(DATA_CFG, raw_dir=str(tmp_path), processed_dir=str(tmp_path))
    return prepared, build_splits(
        prepared.interactions,
        n_users=prepared.n_users,
        n_items=prepared.n_items,
        strategy="leave_one_out",
    )


def test_prepared_ids_are_contiguous(splits):
    prepared, _ = splits
    users = prepared.interactions["user_id"]
    items = prepared.interactions["item_id"]

    assert users.min() == 0 and users.max() == prepared.n_users - 1
    assert items.min() == 0 and items.max() == prepared.n_items - 1


def test_preparation_is_cached(tmp_path):
    first = prepare(DATA_CFG, raw_dir=str(tmp_path), processed_dir=str(tmp_path))
    second = prepare(DATA_CFG, raw_dir=str(tmp_path), processed_dir=str(tmp_path))
    pd.testing.assert_frame_equal(first.interactions, second.interactions)


def test_popularity_runs_end_to_end(splits):
    _, split = splits
    assert leaked_pairs(split) == 0

    model = MODELS.build("popularity")
    model.fit(InteractionMatrix(split.train, split.n_users, split.n_items), TrainContext(run_dir=None))
    metrics = evaluate_model(model, split, ks=[10])

    assert 0.0 < metrics["ndcg@10"] <= 1.0
    assert metrics["n_evaluated_users"] == split.test["user_id"].nunique()


def test_seen_items_are_not_recommended(splits):
    _, split = splits
    model = MODELS.build("popularity")
    train_matrix = InteractionMatrix(split.train, split.n_users, split.n_items)
    model.fit(train_matrix, TrainContext(run_dir=None))

    users = np.array([0, 1, 2])
    ranked = model.recommend(users, k=10, exclude=train_matrix)

    for row, user in enumerate(users):
        assert not set(ranked[row]) & train_matrix.positives[user]


def test_predictions_contract_matches_direct_evaluation(splits):
    """A predictions file scores the same as the model that produced it."""
    _, split = splits
    model = MODELS.build("popularity")
    train_matrix = InteractionMatrix(split.train, split.n_users, split.n_items)
    model.fit(train_matrix, TrainContext(run_dir=None))

    direct = evaluate_model(model, split, ks=[10])

    users = np.array(sorted(split.test["user_id"].unique()))
    ranked = model.recommend(users, 10, train_matrix)
    frame = pd.DataFrame(
        {
            "user_id": np.repeat(users, ranked.shape[1]),
            "item_id": ranked.reshape(-1),
            "rank": np.tile(np.arange(ranked.shape[1]), len(users)),
        }
    )
    via_file = evaluate_predictions(frame, split, ks=[10])

    assert via_file["ndcg@10"] == pytest.approx(direct["ndcg@10"])
    assert via_file["recall@10"] == pytest.approx(direct["recall@10"])


def test_bprmf_beats_popularity(splits, tmp_path):
    """The synthetic data has learnable structure, so a fitted model should exploit it."""
    pytest.importorskip("torch")
    _, split = splits
    train_matrix = InteractionMatrix(split.train, split.n_users, split.n_items)

    baseline = MODELS.build("popularity")
    baseline.fit(train_matrix, TrainContext(run_dir=tmp_path))
    baseline_score = evaluate_model(baseline, split, ks=[10])["ndcg@10"]

    model = MODELS.build("bprmf", embedding_dim=32, learning_rate=0.05)
    model.fit(train_matrix, TrainContext(run_dir=tmp_path, epochs=60, log_every=0, device="cpu"))
    model_score = evaluate_model(model, split, ks=[10])["ndcg@10"]

    assert model_score > baseline_score
