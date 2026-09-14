"""Raw dataset -> filtered, id-remapped, cached interaction table.

The cache is keyed on a hash of the data config, so two jobs with the same data settings
reuse one preprocessing run and a changed setting silently produces a new cache instead
of a stale one. Preprocessing on every job would waste GPU time on work that never
changes.

After this step, ``user_id`` and ``item_id`` are contiguous integers in ``[0, n)`` --
downstream code can use them directly as array indices. The original identifiers are
kept alongside so results can be mapped back.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from recpipe.data import ingest as _ingest  # noqa: F401  (registers the ingests)
from recpipe.data import schema
from recpipe.registry import INGESTS


@dataclass
class PreparedData:
    """Interactions with contiguous ids, plus the mapping back to the originals."""

    name: str
    interactions: pd.DataFrame
    user_ids: np.ndarray
    item_ids: np.ndarray

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def describe(self) -> dict[str, Any]:
        n = len(self.interactions)
        return {
            "dataset": self.name,
            "n_users": self.n_users,
            "n_items": self.n_items,
            "n_interactions": n,
            "density": round(n / (self.n_users * self.n_items), 6),
        }


def config_hash(cfg: Mapping[str, Any]) -> str:
    """Stable short hash of a data config, used as the cache key."""
    payload = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def _apply_kcore(frame: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """Drop rare users and items until both thresholds hold at once."""
    for _ in range(20):
        before = len(frame)
        if min_item > 1:
            counts = frame[schema.ITEM].value_counts()
            frame = frame[frame[schema.ITEM].isin(counts.index[counts >= min_item])]
        if min_user > 1:
            counts = frame[schema.USER].value_counts()
            frame = frame[frame[schema.USER].isin(counts.index[counts >= min_user])]
        if len(frame) == before:
            break
    if frame.empty:
        raise ValueError(
            f"filtering removed every interaction "
            f"(min_user_interactions={min_user}, min_item_interactions={min_item}). "
            "Lower the thresholds or check the ingest."
        )
    return frame


def _remap(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    user_ids = np.sort(frame[schema.USER].unique())
    item_ids = np.sort(frame[schema.ITEM].unique())

    user_index = pd.Series(np.arange(len(user_ids)), index=user_ids)
    item_index = pd.Series(np.arange(len(item_ids)), index=item_ids)

    out = frame.copy()
    out[schema.USER] = out[schema.USER].map(user_index).astype("int64")
    out[schema.ITEM] = out[schema.ITEM].map(item_index).astype("int64")
    return out, user_ids, item_ids


def prepare(
    data_cfg: Mapping[str, Any],
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    force: bool = False,
) -> PreparedData:
    name = data_cfg["name"]
    key = config_hash(data_cfg)
    cache = Path(processed_dir) / f"{name}-{key}"

    if cache.exists() and not force:
        frame = pd.read_parquet(cache / "interactions.parquet")
        users = np.load(cache / "user_ids.npy", allow_pickle=True)
        items = np.load(cache / "item_ids.npy", allow_pickle=True)
        return PreparedData(name=name, interactions=frame, user_ids=users, item_ids=items)

    params = dict(data_cfg.get("params") or {})
    frame = INGESTS.build(data_cfg["ingest"], raw_dir=raw_dir, **params)
    frame = schema.validate(frame, source=f"ingest {data_cfg['ingest']!r}")

    filters = dict(data_cfg.get("filter") or {})
    if filters.get("deduplicate", True):
        # Keep the most recent interaction per (user, item); repeated views would
        # otherwise leak a "future" copy of a held-out item into the training split.
        frame = (
            frame.sort_values(schema.TIMESTAMP)
            .drop_duplicates(subset=[schema.USER, schema.ITEM], keep="last")
        )

    frame = _apply_kcore(
        frame,
        min_user=int(filters.get("min_user_interactions", 1)),
        min_item=int(filters.get("min_item_interactions", 1)),
    )

    frame, user_ids, item_ids = _remap(frame)
    frame = frame.sort_values([schema.USER, schema.TIMESTAMP]).reset_index(drop=True)

    cache.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache / "interactions.parquet", index=False)
    np.save(cache / "user_ids.npy", user_ids)
    np.save(cache / "item_ids.npy", item_ids)

    prepared = PreparedData(name=name, interactions=frame, user_ids=user_ids, item_ids=item_ids)
    (cache / "meta.json").write_text(
        json.dumps({"config": dict(data_cfg), **prepared.describe()}, indent=2, default=str)
    )
    return prepared


def _cli() -> None:
    """Build a cache ahead of time: `python -m recpipe.data.prepare data=ml100k`.

    Useful on a front-end node, where there is internet for downloads.
    """
    import hydra
    from omegaconf import DictConfig, OmegaConf

    @hydra.main(version_base=None, config_path="../../../conf", config_name="config")
    def main(cfg: DictConfig) -> None:
        prepared = prepare(
            OmegaConf.to_container(cfg.data, resolve=True),
            raw_dir=cfg.paths.raw_dir,
            processed_dir=cfg.paths.processed_dir,
            force=bool(cfg.get("force_prepare", False)),
        )
        print(json.dumps(prepared.describe(), indent=2))

    main()


if __name__ == "__main__":
    _cli()
