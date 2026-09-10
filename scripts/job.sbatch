#!/bin/bash
# Generic AI-LAB job wrapper. Resources come from `scripts/submit.py` as sbatch flags,
# which override the defaults below, so this file rarely needs editing.
#
#   sbatch scripts/job.sbatch model=bprmf data=ml100k
#
#SBATCH --job-name=recpipe
#SBATCH --output=logs/%j-%x.out
#SBATCH --error=logs/%j-%x.err
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1

set -euo pipefail

: "${RECPIPE_PROJECT_ROOT:=${SLURM_SUBMIT_DIR:-$PWD}}"
source "$RECPIPE_PROJECT_ROOT/scripts/env.sh"

: "${RECPIPE_ENTRYPOINT:=recpipe.train}"

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

echo "node       : $(hostname)"
echo "job        : ${SLURM_JOB_ID:-none}"
echo "container  : $RECPIPE_CONTAINER"
echo "venv       : $RECPIPE_VENV"
echo "entrypoint : $RECPIPE_ENTRYPOINT"
echo "overrides  : $*"
echo

# --nv only makes sense when the job actually holds a GPU.
NV_FLAG=""
if [ -n "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}${GPU_DEVICE_ORDINAL:-}" ]; then
    NV_FLAG="--nv"
fi

srun singularity exec $NV_FLAG "$RECPIPE_CONTAINER" bash -c \
    "source '$RECPIPE_VENV/bin/activate' && cd '$PROJECT_ROOT' && python -m $RECPIPE_ENTRYPOINT $*"

# Long runs that need more than the 12 hour ceiling: add a SIGTERM trap here plus
# `#SBATCH --signal=B:SIGTERM@60` and `#SBATCH --requeue`, and call `scontrol requeue
# $SLURM_JOB_ID` from the handler. The model side already resumes from
# `<run_dir>/checkpoint.pt`, so only this script needs the change.
