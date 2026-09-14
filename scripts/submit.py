"""Turn a config into a submitted Slurm job.

    python scripts/submit.py model=bprmf data=ml100k slurm=default
    python scripts/submit.py model=bprmf --entrypoint recpipe.export_split --dry-run

Anything that looks like `key=value` is passed straight through to Hydra; everything
starting with `--` is a flag for this script. Resources come from the chosen `slurm`
profile in `conf/slurm/`, so job size is part of the experiment config rather than
something remembered by hand at the prompt.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = PROJECT_ROOT / "conf"

# AI-LAB fair usage, encoded so a bad request fails here instead of in the queue.
MAX_GPUS_PER_JOB = 4
MAX_HOURS = 12


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--entrypoint", default="recpipe.train", help="module to run")
    parser.add_argument("--name", default=None, help="job name (default: <model>-<data>)")
    parser.add_argument("--dry-run", action="store_true", help="print the command, submit nothing")
    known, rest = parser.parse_known_args(argv)

    overrides = [arg for arg in rest if not arg.startswith("--")]
    unknown = [arg for arg in rest if arg.startswith("--")]
    if unknown:
        parser.error(f"unknown flag(s): {' '.join(unknown)}")
    return known, overrides


def hours(time_string: str) -> float:
    """Slurm time string to hours. Accepts HH:MM:SS and D-HH:MM:SS."""
    days = 0
    if "-" in time_string:
        day_part, time_string = time_string.split("-", 1)
        days = int(day_part)
    parts = [int(p) for p in time_string.split(":")]
    while len(parts) < 3:
        parts.append(0)
    return days * 24 + parts[0] + parts[1] / 60 + parts[2] / 3600


def main(argv: list[str]) -> int:
    args, overrides = parse_args(argv)

    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    gpus = int(cfg.slurm.gpus)
    if gpus > MAX_GPUS_PER_JOB:
        print(
            f"slurm.gpus={gpus} exceeds the AI-LAB limit of {MAX_GPUS_PER_JOB} GPUs per job.",
            file=sys.stderr,
        )
        return 1

    requested_hours = hours(str(cfg.slurm.time))
    if requested_hours > MAX_HOURS:
        print(
            f"slurm.time={cfg.slurm.time} exceeds the AI-LAB maximum of {MAX_HOURS} hours. "
            "Use checkpointing and requeue instead of a longer wall clock.",
            file=sys.stderr,
        )
        return 1

    name = args.name or f"{cfg.model.name}-{cfg.data.name}"
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)

    command = [
        "sbatch",
        f"--job-name={name}",
        f"--time={cfg.slurm.time}",
        f"--mem={cfg.slurm.mem}",
        f"--cpus-per-task={cfg.slurm.cpus_per_task}",
        f"--output={PROJECT_ROOT}/logs/%j-{name}.out",
        f"--error={PROJECT_ROOT}/logs/%j-{name}.err",
        f"--export=ALL,RECPIPE_PROJECT_ROOT={PROJECT_ROOT},RECPIPE_ENTRYPOINT={args.entrypoint}",
    ]
    if gpus > 0:
        command.append(f"--gres=gpu:{gpus}")
    command.append(str(PROJECT_ROOT / "scripts" / "job.sbatch"))
    command.extend(overrides)

    printable = " ".join(command)
    if args.dry_run:
        print(printable)
        return 0

    print(printable)
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=os.environ.copy())
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
