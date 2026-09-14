"""Score a predictions file produced outside this pipeline (integration level 3).

    python -m recpipe.evaluate data=ml100k predictions=/path/to/predictions.parquet

The data and task configs must match the ones used for `recpipe.export_split`, otherwise
the split, and therefore the ground truth, is a different one.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from recpipe.data.prepare import prepare
from recpipe.data.splits import build_splits
from recpipe.eval.evaluator import evaluate_predictions
from recpipe.utils.runtime import write_json


def _plain(node: object) -> dict:
    if node is None:
        return {}
    return OmegaConf.to_container(node, resolve=True) or {}


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if not cfg.predictions:
        raise SystemExit("set predictions=/path/to/predictions.parquet")

    path = Path(cfg.predictions)
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

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

    metrics = evaluate_predictions(frame, splits, ks=[int(k) for k in cfg.task.eval.ks])

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    record = {
        "run_dir": str(run_dir),
        "model": f"external:{path.stem}",
        "data": cfg.data.name,
        "task": cfg.task.name,
        "trainer": "external",
        "predictions": str(path),
        "dataset": prepared.describe(),
        "splits": splits.describe(),
        "metrics": metrics,
    }
    write_json(run_dir / "metrics.json", record)
    print({name: round(value, 4) for name, value in metrics.items()})
    print(f"written to {run_dir}")


if __name__ == "__main__":
    main()
