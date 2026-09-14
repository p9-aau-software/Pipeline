"""Recommend whatever is most popular, ignoring who is asking.

Not decoration. Its numbers are predictable, so when a split or a metric is wrong this
baseline is usually what shows it -- and any real model that cannot beat it is not
working, however good its loss curve looks.
"""

from __future__ import annotations

import numpy as np

from recpipe.data.dataset import InteractionMatrix
from recpipe.models.base import RecModel, TrainContext, exclusion_lists, top_k
from recpipe.registry import MODELS


@MODELS.register("popularity")
class PopularityModel(RecModel):
    def __init__(self, **_: object) -> None:
        self._scores: np.ndarray | None = None

    def fit(self, train: InteractionMatrix, ctx: TrainContext) -> None:
        self._scores = train.item_popularity()

    def recommend(
        self, users: np.ndarray, k: int, exclude: InteractionMatrix | None = None
    ) -> np.ndarray:
        if self._scores is None:
            raise RuntimeError("PopularityModel.recommend called before fit")
        scores = np.broadcast_to(self._scores, (len(users), len(self._scores)))
        return top_k(scores, k, exclusion_lists(users, exclude))
