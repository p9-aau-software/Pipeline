"""Collect every run under `outputs/` into one leaderboard.

    python scripts/compare.py
    python scripts/compare.py --data ml100k --sort ndcg@10
    python scripts/compare.py --csv results.csv

Each run writes its own `metrics.json`; this just gathers them. Nothing to keep in sync,
and a deleted run simply disappears from the table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_COLUMNS = ["model", "data", "trainer", "device", "train_seconds"]


def load_runs(outputs: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(outputs.rglob("metrics.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"skipping unreadable {path}")
            continue

        row = {key: record.get(key) for key in BASE_COLUMNS}
        row["run"] = str(Path(record.get("run_dir", path.parent)).name)
        row["job"] = record.get("slurm_job_id")
        row.update(record.get("metrics", {}))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--sort", default="ndcg@10", help="metric column to rank by")
    parser.add_argument("--data", default=None, help="only runs on this dataset")
    parser.add_argument("--model", default=None, help="only runs of this model")
    parser.add_argument("--csv", default=None, help="also write the table here")
    args = parser.parse_args()

    frame = load_runs(Path(args.outputs))
    if frame.empty:
        print(f"no runs found under {args.outputs}")
        return 0

    if args.data:
        frame = frame[frame["data"] == args.data]
    if args.model:
        frame = frame[frame["model"] == args.model]
    if frame.empty:
        print("no runs match those filters")
        return 0

    if args.sort in frame.columns:
        frame = frame.sort_values(args.sort, ascending=False)
    else:
        print(f"note: {args.sort!r} is not in these runs; leaving the order alone")

    # Drop metric columns that no run reported, so the table stays readable.
    frame = frame.dropna(axis=1, how="all")

    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if args.csv:
        frame.to_csv(args.csv, index=False)
        print(f"\nwritten to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
