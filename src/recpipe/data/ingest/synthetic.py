"""A small generated dataset. The backbone of every test and smoke run.

The structure is deliberately *learnable*: users have sparse preferences over latent
topics and items belong to topics, so a factorisation model can beat a popularity
baseline. A purely random dataset would make the pipeline look broken when it is fine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recpipe.data import schema
from recpipe.registry import INGESTS


@INGESTS.register("synthetic")
def synthetic(
    n_users: int = 500,
    n_items: int = 200,
    n_topics: int = 8,
    interactions_per_user: int = 40,
    min_interactions: int = 6,
    popularity_skew: float = 0.8,
    seed: int = 0,
    **_: object,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    item_topic = rng.integers(0, n_topics, size=n_items)
    # Long tail: a shuffled rank -> 1/rank^skew popularity curve.
    rank = rng.permutation(n_items) + 1.0
    popularity = rank ** (-popularity_skew)

    users: list[np.ndarray] = []
    items: list[np.ndarray] = []
    times: list[np.ndarray] = []

    for user in range(n_users):
        # Concentration below 1 gives each user a few strong topics, not a flat taste.
        topic_pref = rng.dirichlet(np.full(n_topics, 0.3))
        probs = topic_pref[item_topic] * popularity
        probs /= probs.sum()

        count = int(rng.poisson(interactions_per_user))
        count = max(min_interactions, min(count, n_items))

        chosen = rng.choice(n_items, size=count, replace=False, p=probs)
        start = int(rng.integers(1_500_000_000, 1_600_000_000))
        offsets = np.cumsum(rng.integers(60, 86_400, size=count))

        users.append(np.full(count, user))
        items.append(chosen)
        times.append(start + offsets)

    frame = pd.DataFrame(
        {
            schema.USER: np.concatenate(users),
            schema.ITEM: np.concatenate(items),
            schema.TIMESTAMP: np.concatenate(times),
            schema.WEIGHT: 1.0,
        }
    )
    return schema.validate(frame, source="synthetic ingest")
