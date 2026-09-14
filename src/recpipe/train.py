"""Train and evaluate one configuration.

    python -m recpipe.train                                  # synthetic + popularity
    python -m recpipe.train model=bprmf data=ml100k trainer=default
    python -m recpipe.train model=bprmf model.params.embedding_dim=128

Every run writes its own directory under `outputs/` holding the resolved config, the
metrics and any checkpoints, so a result can always be traced back to what produced it.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from recpipe import models as _models  # noqa: F401  (registers the models)
from recpipe.data.dataset import InteractionMatrix
from recpipe.data.prepare import prepare
from recpipe.data.splits import build_splits, leaked_pairs
from recpipe.eval.evaluator import evaluate_model
from recpipe.models.base import TrainContext
from recpipe.registry import MODELS
from recpipe.utils.runtime import set_seed, write_json


def _plain(node: object) -> dict:
    """OmegaConf node to a plain dict, tolerating a missing or null node."""
    if node is None:
        return {}
    return OmegaConf.to_container(node, resolve=True) or {}


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    set_seed(int(cfg.trainer.seed))

    prepared = prepare(
        _plain(cfg.data),
        raw_dir=cfg.paths.raw_dir,
        processed_dir=cfg.paths.processed_dir,
        force=bool(cfg.force_prepare),
    )
    print(f"data: {prepared.describe()}")

    split_cfg = cfg.task.split
    splits = build_splits(
        prepared.interactions,
        n_users=prepared.n_users,
        n_items=prepared.n_items,
        strategy=split_cfg.strategy,
        drop_unseen=bool(split_cfg.drop_unseen),
        **_plain(split_cfg.get("params")),
    )
    print(f"splits: {splits.describe()}")

    model = MODELS.build(cfg.model.name, **_plain(cfg.model.get("params")))

    context = TrainContext(
        run_dir=run_dir,
        device=cfg.trainer.device,
        seed=int(cfg.trainer.seed),
        epochs=int(cfg.trainer.epochs),
        log_every=int(cfg.trainer.log_every),
        checkpoint_every=int(cfg.trainer.checkpoint_every),
    )

    started = time.time()
    model.fit(InteractionMatrix(splits.train, splits.n_users, splits.n_items), context)
    train_seconds = time.time() - started

    ks = [int(k) for k in cfg.task.eval.ks]
    eval_kwargs = {
        "ks": ks,
        "exclude_seen": bool(cfg.task.eval.exclude_seen),
        "batch_size": int(cfg.task.eval.batch_size),
    }
    test_metrics = evaluate_model(model, splits, target="test", **eval_kwargs)
    val_metrics = (
        evaluate_model(model, splits, target="val", **eval_kwargs)
        if not splits.val.empty
        else {}
    )

    record = {
        "run_dir": str(run_dir),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "model": cfg.model.name,
        "data": cfg.data.name,
        "task": cfg.task.name,
        "trainer": cfg.trainer.name,
        "device": context.device,
        "train_seconds": round(train_seconds, 2),
        "dataset": prepared.describe(),
        "splits": splits.describe(),
        "leaked_pairs": leaked_pairs(splits),
        "params": _plain(cfg.model.get("params")),
        "metrics": test_metrics,
        "val_metrics": val_metrics,
    }

    write_json(run_dir / "metrics.json", record)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    model.save(run_dir / "model.pt")

    headline = ", ".join(f"{k}={v:.4f}" for k, v in sorted(test_metrics.items()))
    print(f"\n{cfg.model.name} on {cfg.data.name}: {headline}")
    print(f"written to {run_dir}")


if __name__ == "__main__":
    main()
