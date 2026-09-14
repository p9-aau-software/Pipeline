"""The canonical interaction format.

Every dataset is normalised into this one table before anything else touches it. That
is what keeps "add a dataset" at roughly thirty lines: an ingest function only has to
produce these four columns, and splits, sampling, training and evaluation are shared
from there on.

    user_id    original user identifier (any hashable type)
    item_id    original item identifier (any hashable type)
    timestamp  int64, seconds since epoch or any monotonic ordering
    weight     float32, interaction strength (1.0 for plain implicit feedback)
"""

from __future__ import annotations

import pandas as pd

USER = "user_id"
ITEM = "item_id"
TIMESTAMP = "timestamp"
WEIGHT = "weight"

COLUMNS = (USER, ITEM, TIMESTAMP, WEIGHT)


def validate(df: pd.DataFrame, source: str = "ingest") -> pd.DataFrame:
    """Check and normalise a raw ingest result.

    Returns a copy with exactly the canonical columns, in order, with fixed dtypes.
    Raises ValueError with an actionable message when an ingest is malformed -- these
    errors are cheap to hit and expensive to debug later, so they are strict.
    """
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{source} produced a table missing required column(s) {missing}. "
            f"Got columns: {list(df.columns)}. Expected: {list(COLUMNS)}."
        )

    out = df.loc[:, list(COLUMNS)].copy()

    for column in COLUMNS:
        if out[column].isna().any():
            raise ValueError(f"{source} produced nulls in column {column!r}")

    try:
        out[TIMESTAMP] = out[TIMESTAMP].astype("int64")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: {TIMESTAMP} must be integer-like: {exc}") from exc

    try:
        out[WEIGHT] = out[WEIGHT].astype("float32")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: {WEIGHT} must be numeric: {exc}") from exc

    if out.empty:
        raise ValueError(f"{source} produced zero interactions")

    return out.reset_index(drop=True)
