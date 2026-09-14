"""In-memory structures models train against, plus negative sampling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recpipe.data import schema


class InteractionMatrix:
    """A user-item interaction set with the lookups models and evaluation need."""

    def __init__(self, frame: pd.DataFrame, n_users: int, n_items: int) -> None:
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.users = frame[schema.USER].to_numpy(dtype=np.int64)
        self.items = frame[schema.ITEM].to_numpy(dtype=np.int64)
        self.weights = frame[schema.WEIGHT].to_numpy(dtype=np.float32)
        self._positives: list[set[int]] | None = None

    def __len__(self) -> int:
        return len(self.users)

    @property
    def positives(self) -> list[set[int]]:
        """Per-user set of interacted items. Built once, on first use."""
        if self._positives is None:
            sets: list[set[int]] = [set() for _ in range(self.n_users)]
            for user, item in zip(self.users, self.items):
                sets[user].add(int(item))
            self._positives = sets
        return self._positives

    def item_popularity(self) -> np.ndarray:
        return np.bincount(self.items, minlength=self.n_items).astype(np.float64)

    def active_users(self) -> np.ndarray:
        return np.unique(self.users)


class BPRSampler:
    """Uniform negative sampling for pairwise ranking losses.

    Rejection is done with a handful of vectorised retries rather than a strict loop:
    on any realistic catalogue a collision is rare, and a leftover false negative is
    ordinary label noise, not a correctness bug.
    """

    def __init__(self, matrix: InteractionMatrix, seed: int = 0, max_retries: int = 3) -> None:
        self.matrix = matrix
        self.rng = np.random.default_rng(seed)
        self.max_retries = max_retries

    def batches(self, batch_size: int) -> "list[tuple[np.ndarray, np.ndarray, np.ndarray]]":
        """One epoch: every interaction used once, in random order."""
        order = self.rng.permutation(len(self.matrix))
        out = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            users = self.matrix.users[idx]
            positive = self.matrix.items[idx]
            out.append((users, positive, self._negatives(users)))
        return out

    def _negatives(self, users: np.ndarray) -> np.ndarray:
        positives = self.matrix.positives
        negatives = self.rng.integers(0, self.matrix.n_items, size=len(users))
        for _ in range(self.max_retries):
            clash = np.fromiter(
                (item in positives[user] for user, item in zip(users, negatives)),
                dtype=bool,
                count=len(users),
            )
            if not clash.any():
                break
            negatives[clash] = self.rng.integers(0, self.matrix.n_items, size=int(clash.sum()))
        return negatives


def ground_truth(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    """Held-out items per user, for evaluation."""
    grouped = frame.groupby(schema.USER)[schema.ITEM].apply(lambda s: s.to_numpy(dtype=np.int64))
    return {int(user): items for user, items in grouped.items()}
