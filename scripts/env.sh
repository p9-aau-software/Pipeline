#!/bin/bash
# Shared environment settings for AI-LAB. Source this file, do not execute it.
#
# Override anything by exporting it before sourcing, e.g.
#   export RECPIPE_CONTAINER=/ceph/container/pytorch/pytorch_24.09.sif

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Newest PyTorch container in the shared image directory unless told otherwise.
if [ -z "${RECPIPE_CONTAINER:-}" ]; then
    RECPIPE_CONTAINER="$(ls -1 /ceph/container/pytorch/*.sif 2>/dev/null | sort -V | tail -1)"
fi

# The venv lives outside the repo so `git clean` and branch switches cannot destroy it.
: "${RECPIPE_VENV:=$HOME/venvs/recpipe}"

export PROJECT_ROOT RECPIPE_CONTAINER RECPIPE_VENV
