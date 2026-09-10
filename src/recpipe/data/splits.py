"""Train/validation/test splitting.

Splitting lives here, once, on purpose: two models are only comparable if they were
scored on the same held-out interactions. A model that brings its own split is not a
result, it is a coincidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from recpipe.data import schema
from recpipe.registry import Registry

SPLITS = Registry("split strategy")


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    n_users: int
    n_items: int

    def describe(self) -> dict[str, int]:
        return {
            "n_train": len(self.train),
            "n_val": len(self.val),
            "n_test": len(self.test),
            "n_test_users": int(self.test[schema.USER].nunique()),
        }


@SPLITS.register("leave_one_out")
def leave_one_out(frame: pd.DataFrame, min_interactions: int = 3, **_: object) -> tuple[pd.DataFrame, ...]:
    """Newest interaction per user to test, second newest to validation.

    The standard protocol for implicit top-K. Users with too little history stay wholly
    in train rather than being evaluated on a single noisy point.
    """
    frame = frame.sort_values([schema.USER, schema.TIMESTAMP])
    rank_from_end = frame.groupby(schema.USER).cumcount(ascending=False)
    eligible = frame.groupby(schema.USER)[schema.ITEM].transform("size") >= min_interactions

    test = frame[eligible & (rank_from_end == 0)]
    val = frame[eligible & (rank_from_end == 1)]
    train = frame[(~eligible) | (rank_from_end >= 2)]
    return train, val, test


@SPLITS.register("temporal")
def temporal(
    frame: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    **_: object,
) -> tuple[pd.DataFrame, ...]:
    """Global time cut. Closer to deployment: predict the future from the past."""
    cut_train = frame[schema.TIMESTAMP].quantile(train_ratio)
    cut_val = frame[schema.TIMESTAMP].quantile(train_ratio + val_ratio)

    train = frame[frame[schema.TIMESTAMP] <= cut_train]
    val = frame[(frame[schema.TIMESTAMP] > cut_train) & (frame[schema.TIMESTAMP] <= cut_val)]
    test = frame[frame[schema.TIMESTAMP] > cut_val]
    return train, val, test


def _drop_unseen(part: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Remove cold-start users and items -- an id-indexed model cannot score them."""
    known_users = set(train[schema.USER].unique())
    known_items = set(train[schema.ITEM].unique())
    return part[part[schema.USER].isin(known_users) & part[schema.ITEM].isin(known_items)]


def build_splits(
    frame: pd.DataFrame,
    n_users: int,
    n_items: int,
    strategy: str = "leave_one_out",
    drop_unseen: bool = True,
    **params: object,
) -> Splits:
    train, val, test = SPLITS.build(strategy, frame=frame, **params)

    if drop_unseen:
        val = _drop_unseen(val, train)
        test = _drop_unseen(test, train)

    if train.empty:
        raise ValueError(f"split {strategy!r} produced an empty training set")
    if test.empty:
        raise ValueError(
            f"split {strategy!r} produced an empty test set. With a temporal split this "
            "usually means the time cut is too late, or every test user is cold-start."
        )

    return Splits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
        n_users=n_users,
        n_items=n_items,
    )


def leaked_pairs(splits: Splits) -> int:
    """Count held-out (user, item) pairs that also appear in train. Must be zero."""
    train_pairs = set(map(tuple, splits.train[[schema.USER, schema.ITEM]].to_numpy()))
    held_out = pd.concat([splits.val, splits.test])
    held_pairs = set(map(tuple, held_out[[schema.USER, schema.ITEM]].to_numpy()))
    return len(train_pairs & held_pairs)
