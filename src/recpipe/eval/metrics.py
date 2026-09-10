"""Top-K ranking metrics.

Pure functions over a ranked-item array, so they can be unit tested against numbers
worked out by hand -- which is the only way to be sure an evaluation is right.

    ranked    (n_users, k) item indices, best first
    relevant  list of arrays, the held-out items for each of those users, same order
"""

from __future__ import annotations

import numpy as np


def _hit_matrix(ranked: np.ndarray, relevant: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if len(ranked) != len(relevant):
        raise ValueError(f"got {len(ranked)} ranking rows but {len(relevant)} ground-truth rows")

    hits = np.zeros(ranked.shape, dtype=bool)
    counts = np.zeros(len(ranked), dtype=np.float64)
    for row, items in enumerate(relevant):
        truth = set(np.asarray(items).tolist())
        counts[row] = len(truth)
        hits[row] = [int(item) in truth for item in ranked[row]]
    return hits, counts


def ranking_metrics(
    ranked: np.ndarray,
    relevant: list[np.ndarray],
    ks: "list[int] | tuple[int, ...]" = (10,),
    n_items: int | None = None,
) -> dict[str, float]:
    ranked = np.asarray(ranked)
    hits, n_relevant = _hit_matrix(ranked, relevant)
    safe_relevant = np.maximum(n_relevant, 1.0)

    results: dict[str, float] = {}
    for k in ks:
        k = min(int(k), ranked.shape[1])
        window = hits[:, :k]
        n_hit = window.sum(axis=1).astype(np.float64)

        discounts = 1.0 / np.log2(np.arange(k) + 2.0)
        dcg = (window * discounts).sum(axis=1)
        # Ideal DCG: all relevant items packed into the top positions.
        ideal_length = np.minimum(n_relevant, k).astype(int)
        cumulative = np.concatenate([[0.0], np.cumsum(discounts)])
        idcg = cumulative[ideal_length]

        first_hit = np.argmax(window, axis=1)
        has_hit = window.any(axis=1)
        reciprocal = np.where(has_hit, 1.0 / (first_hit + 1.0), 0.0)

        results[f"recall@{k}"] = float(np.mean(n_hit / safe_relevant))
        results[f"precision@{k}"] = float(np.mean(n_hit / k))
        results[f"ndcg@{k}"] = float(np.mean(np.divide(dcg, np.maximum(idcg, 1e-12))))
        results[f"hit_rate@{k}"] = float(np.mean(has_hit))
        results[f"mrr@{k}"] = float(np.mean(reciprocal))
        if n_items:
            # Padding for short rankings is -1; it is not an item and must not be counted.
            recommended = ranked[:, :k]
            distinct = np.unique(recommended[recommended >= 0])
            results[f"coverage@{k}"] = float(len(distinct) / n_items)

    return results
