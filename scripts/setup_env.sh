#!/bin/bash
# One-time setup: create a venv inside the PyTorch container, on /ceph.
#
# Why a venv and not `singularity build --sandbox`: adding a package takes seconds
# instead of minutes, and a hostile external repo can get its own venv (RECPIPE_VENV=...)
# without touching this one. The container still supplies torch + CUDA.
#
#   bash scripts/setup_env.sh
set -euo pipefail

source "$(dirname "$0")/env.sh"

if [ -z "$RECPIPE_CONTAINER" ] || [ ! -f "$RECPIPE_CONTAINER" ]; then
    echo "No container found. Check what is available:" >&2
    echo "    ls /ceph/container" >&2
    echo "then: export RECPIPE_CONTAINER=/ceph/container/<...>.sif" >&2
    exit 1
fi

echo "Container : $RECPIPE_CONTAINER"
echo "Venv      : $RECPIPE_VENV"
echo "Project   : $PROJECT_ROOT"

# Build on a compute node rather than the front-end (front-ends are for preparation only).
srun --mem=8G --cpus-per-task=4 --time=00:30:00 \
    singularity exec "$RECPIPE_CONTAINER" bash -c "
        set -euo pipefail
        python3 -m venv --system-site-packages '$RECPIPE_VENV'
        source '$RECPIPE_VENV/bin/activate'
        pip install --upgrade pip
        pip install -e '$PROJECT_ROOT[dev]'
    "

echo
echo "Done. Verify GPU access with:"
echo "    srun --gres=gpu:1 --time=00:05:00 singularity exec --nv \$RECPIPE_CONTAINER \\"
echo "        bash -c \"source $RECPIPE_VENV/bin/activate && python -c 'import torch; print(torch.cuda.is_available())'\""
