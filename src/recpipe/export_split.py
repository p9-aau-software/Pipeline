"""Write the current split to disk so an outside repo can train on exactly our data.

    python -m recpipe.export_split data=ml100k export_dir=exports/ml100k

Hand the resulting directory to the foreign code, let it train however it likes, and have
it write `predictions.parquet` with columns `user_id, item_id, rank` using the same
integer indices. Score it with `python -m recpipe.evaluate`.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from recpipe.data.prepare import prepare
from recpipe.data.splits import build_splits
from recpipe.utils.runtime import write_json


def _plain(node: object) -> dict:
    if node is None:
        return {}
    return OmegaConf.to_container(node, resolve=True) or {}


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    target = Path(cfg.export_dir or f"exports/{cfg.data.name}")
    target.mkdir(parents=True, exist_ok=True)

    prepared = prepare(
        _plain(cfg.data),
        raw_dir=cfg.paths.raw_dir,
        processed_dir=cfg.paths.processed_dir,
    )
    split_cfg = cfg.task.split
    splits = build_splits(
        prepared.interactions,
        n_users=prepared.n_users,
        n_items=prepared.n_items,
        strategy=split_cfg.strategy,
        drop_unseen=bool(split_cfg.drop_unseen),
        **_plain(split_cfg.get("params")),
    )

    # Both formats: parquet for us, CSV because plenty of research code only reads CSV.
    for name in ("train", "val", "test"):
        part = getattr(splits, name)
        part.to_parquet(target / f"{name}.parquet", index=False)
        part.to_csv(target / f"{name}.csv", index=False)

    np.save(target / "user_ids.npy", prepared.user_ids)
    np.save(target / "item_ids.npy", prepared.item_ids)
    write_json(
        target / "meta.json",
        {
            "data": _plain(cfg.data),
            "split": _plain(split_cfg),
            "dataset": prepared.describe(),
            "splits": splits.describe(),
            "note": (
                "user_id/item_id are contiguous indices. Predictions must use these same "
                "indices, with columns user_id, item_id, rank where rank 0 is best."
            ),
        },
    )
    print(f"exported to {target}")


if __name__ == "__main__":
    main()
