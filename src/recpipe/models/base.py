"""What every model must look like, whoever wrote it.

Three integration levels are supported, cheapest first:

1. **Native** -- your own model: subclass ``RecModel``, add ``@MODELS.register("name")``.
2. **Module wrap** -- an outside repo gives you a scorer: wrap it in a ``RecModel`` and
   reuse this pipeline's training loop, splits and evaluation.
3. **Predictions contract** -- an outside repo is too tangled to touch: run it unchanged,
   have it write ``predictions.parquet`` (``user_id, item_id, rank``) and score that with
   ``recpipe.evaluate``.

Level 3 is what makes a foreign repo comparable without rewriting it. See
``models/external/README.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from recpipe.data.dataset import InteractionMatrix


@dataclass
class TrainContext:
    """Everything a model needs from the run that is not a hyperparameter."""

    run_dir: Path | None = None
    device: str = "cpu"
    seed: int = 0
    epochs: int = 10
    log_every: int = 1
    checkpoint_every: int = 0  # 0 disables checkpointing
    extra: dict = field(default_factory=dict)


class RecModel(ABC):
    """Fit on interactions, return ranked items per user."""

    @abstractmethod
    def fit(self, train: InteractionMatrix, ctx: TrainContext) -> None: ...

    @abstractmethod
    def recommend(
        self, users: np.ndarray, k: int, exclude: InteractionMatrix | None = None
    ) -> np.ndarray:
        """Return an ``(len(users), k)`` array of item indices, best first."""

    # Checkpointing is optional: a model with no learned state can ignore both.
    def save(self, path: Path) -> None:  # pragma: no cover - trivial default
        return None

    def load(self, path: Path) -> int:  # pragma: no cover - trivial default
        """Restore state and return the epoch to resume from."""
        return 0


def top_k(scores: np.ndarray, k: int, excluded: list[np.ndarray] | None = None) -> np.ndarray:
    """Top ``k`` columns per row, best first, after masking out excluded items.

    Shared by every model so that tie-breaking and exclusion behave identically -- a
    difference here would show up as a difference in metrics and get blamed on the model.
    """
    scores = np.asarray(scores, dtype=np.float64).copy()
    if excluded is not None:
        for row, items in enumerate(excluded):
            if len(items):
                scores[row, items] = -np.inf

    n_items = scores.shape[1]
    k = min(k, n_items)
    candidates = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[rows, candidates], axis=1)
    return candidates[rows, order]


def exclusion_lists(users: np.ndarray, exclude: InteractionMatrix | None) -> list[np.ndarray] | None:
    """Per-user arrays of items to mask, in the order given by ``users``."""
    if exclude is None:
        return None
    positives = exclude.positives
    return [np.fromiter(positives[int(u)], dtype=np.int64) for u in users]
