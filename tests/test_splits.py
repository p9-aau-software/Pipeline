"""Splitting is the part that quietly invalidates results when it is wrong."""

from __future__ import annotations

import pandas as pd
import pytest

from recpipe.data import schema
from recpipe.data.splits import build_splits, leaked_pairs


def make_frame() -> pd.DataFrame:
    """Three users, five items, each user seeing them in a different order."""
    rows = []
    for user in range(3):
        for step in range(5):
            rows.append(
                {
                    schema.USER: user,
                    schema.ITEM: (user + step) % 5,
                    schema.TIMESTAMP: step,
                    schema.WEIGHT: 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_leave_one_out_holds_out_the_newest_interaction():
    frame = make_frame()
    splits = build_splits(frame, n_users=3, n_items=5, strategy="leave_one_out")

    assert len(splits.test) == 3
    assert len(splits.val) == 3
    assert len(splits.train) == 9

    for user in range(3):
        newest = frame[frame[schema.USER] == user][schema.TIMESTAMP].max()
        held_out = splits.test[splits.test[schema.USER] == user][schema.TIMESTAMP].iloc[0]
        assert held_out == newest


def test_no_held_out_pair_appears_in_train():
    splits = build_splits(make_frame(), n_users=3, n_items=5, strategy="leave_one_out")
    assert leaked_pairs(splits) == 0


def test_users_with_too_little_history_stay_in_train():
    frame = make_frame()
    extra = pd.DataFrame(
        [
            {schema.USER: 3, schema.ITEM: 0, schema.TIMESTAMP: 0, schema.WEIGHT: 1.0},
            {schema.USER: 3, schema.ITEM: 1, schema.TIMESTAMP: 1, schema.WEIGHT: 1.0},
        ]
    )
    splits = build_splits(
        pd.concat([frame, extra]), n_users=4, n_items=5, strategy="leave_one_out"
    )

    assert 3 not in set(splits.test[schema.USER])
    assert (splits.train[schema.USER] == 3).sum() == 2


def test_temporal_split_respects_the_time_cut():
    splits = build_splits(
        make_frame(),
        n_users=3,
        n_items=5,
        strategy="temporal",
        train_ratio=0.6,
        val_ratio=0.2,
    )

    assert splits.train[schema.TIMESTAMP].max() <= splits.test[schema.TIMESTAMP].min()
    assert leaked_pairs(splits) == 0


def test_empty_test_split_is_an_error_not_a_silent_zero():
    frame = make_frame()
    with pytest.raises(ValueError):
        build_splits(frame, n_users=3, n_items=5, strategy="temporal", train_ratio=1.0, val_ratio=0.0)
