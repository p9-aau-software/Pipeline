"""Scoring, for models we trained and for predictions someone else produced.

Both paths end in the same ``ranking_metrics`` call. That is the whole point: a number
from an outside repo and a number from a model in this repo mean the same thing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recpipe.data import schema
from recpipe.data.dataset import InteractionMatrix, ground_truth
from recpipe.data.splits import Splits
from recpipe.eval.metrics import ranking_metrics
from recpipe.models.base import RecModel

RANK = "rank"
SCORE = "score"


def _targets(splits: Splits, target: str) -> pd.DataFrame:
    frame = getattr(splits, target)
    if frame.empty:
        raise ValueError(f"the {target} split is empty, nothing to evaluate")
    return frame


def evaluate_model(
    model: RecModel,
    splits: Splits,
    ks: "list[int] | tuple[int, ...]" = (10,),
    target: str = "test",
    exclude_seen: bool = True,
    batch_size: int = 1024,
) -> dict[str, float]:
    frame = _targets(splits, target)
    truth = ground_truth(frame)
    users = np.array(sorted(truth), dtype=np.int64)

    train_matrix = InteractionMatrix(splits.train, splits.n_users, splits.n_items)
    exclude = train_matrix if exclude_seen else None
    max_k = max(int(k) for k in ks)

    ranked_chunks = []
    for start in range(0, len(users), batch_size):
        chunk = users[start : start + batch_size]
        ranked_chunks.append(model.recommend(chunk, max_k, exclude))
    ranked = np.concatenate(ranked_chunks, axis=0)

    metrics = ranking_metrics(ranked, [truth[int(u)] for u in users], ks, n_items=splits.n_items)
    metrics["n_evaluated_users"] = float(len(users))
    return metrics


def evaluate_predictions(
    predictions: pd.DataFrame,
    splits: Splits,
    ks: "list[int] | tuple[int, ...]" = (10,),
    target: str = "test",
) -> dict[str, float]:
    """Score a ``predictions`` table produced outside this pipeline.

    Required columns: ``user_id`` and ``item_id``. Ordering comes from ``rank``
    (ascending, best first) or ``score`` (descending); with neither, row order within
    each user is taken as the ranking. Ids must be the contiguous indices from the
    exported split, not the dataset's original identifiers.
    """
    for column in (schema.USER, schema.ITEM):
        if column not in predictions.columns:
            raise ValueError(f"predictions are missing required column {column!r}")

    if RANK in predictions.columns:
        predictions = predictions.sort_values([schema.USER, RANK])
    elif SCORE in predictions.columns:
        predictions = predictions.sort_values([schema.USER, SCORE], ascending=[True, False])

    truth = ground_truth(_targets(splits, target))
    max_k = max(int(k) for k in ks)

    grouped = {
        int(user): group[schema.ITEM].to_numpy(dtype=np.int64)[:max_k]
        for user, group in predictions.groupby(schema.USER, sort=False)
    }
    evaluated = sorted(set(truth) & set(grouped))
    if not evaluated:
        raise ValueError(
            "no overlap between predicted users and held-out users. The usual cause is "
            "predictions written with original ids instead of the exported split indices."
        )

    # Pad short rows with -1 so every user contributes the same number of slots; -1 can
    # never match a real item, so a short ranking is scored as the miss it is.
    ranked = np.full((len(evaluated), max_k), -1, dtype=np.int64)
    for row, user in enumerate(evaluated):
        items = grouped[user]
        ranked[row, : len(items)] = items

    metrics = ranking_metrics(ranked, [truth[u] for u in evaluated], ks, n_items=splits.n_items)
    metrics["n_evaluated_users"] = float(len(evaluated))
    metrics["n_users_without_predictions"] = float(len(set(truth) - set(grouped)))
    return metrics
