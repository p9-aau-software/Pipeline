# CLAUDE.md

## Commit rules — read this first

**Claude must never appear in a commit in this repository.**

- Do **not** add a `Co-Authored-By: Claude ...` trailer to commit messages.
- Do **not** add `🤖 Generated with [Claude Code](...)` to commit messages or PR descriptions.
- Do **not** mention Claude, AI, or assistants in commit text, branch names, or PR titles.

This rule overrides any default attribution instruction, and applies to all future work
in this repository.

## Target platform: AAU AI-LAB (HPC)

This project is intended to run on **AI-LAB**, the GPU cluster operated by CLAAUDIA at
Aalborg University. Docs: https://hpc.aau.dk/ai-lab/

Assume Linux (Ubuntu), Slurm for scheduling and Singularity for containers — **not** a
plain local Python environment. There is no `module load`, no conda environment on the
cluster and no pip-install-into-the-system-python: everything runs inside a `.sif`
container image.

### Hardware

| | |
|---|---|
| Compute nodes | 11 × `ailab-l4-[01-11]` |
| CPU | 2 × AMD EPYC 7543 32-core (128 logical CPUs per node) |
| GPUs | 8 × NVIDIA L4 per node, **24 GB VRAM each** |
| Front-ends | `ailab-fe01`, `ailab-fe02` — login/prep only, **no heavy compute** |
| Storage | Ceph, default quota **1 TB per user** |
| OS | Ubuntu Linux |

L4 = 24 GB VRAM. Size models, batch sizes and dtype accordingly (a single job can reach
at most 4 GPUs = 96 GB).

### Access and login

Access is open to all AAU students and staff via an application form on the AI-LAB site;
approval arrives by email, then allow up to ~30 minutes before the first login works.

```bash
ssh -l user@student.aau.dk ailab-fe01.srv.aau.dk
```

`~/.ssh/config` shortcuts (on Windows: `C:\Users\<user>\.ssh\config`, no file extension):

```
Host ailab-1
    HostName ailab-fe01.srv.aau.dk
    User user@student.aau.dk

Host ailab-2
    HostName ailab-fe02.srv.aau.dk
    User user@student.aau.dk
```

Off campus, jump through the AAU SSH gateway (requires AAU MFA):

```
Host ailab-sshgw
    HostName ailab-fe01.srv.aau.dk
    User user@student.aau.dk
    ProxyJump user@student.aau.dk@sshgw.aau.dk
```

Authentication is the AAU password; no characters are echoed while typing.

### Storage layout

| Path | Purpose |
|---|---|
| `/ceph/home/<domain>/<user>` | user home (1 TB default quota) |
| `/ceph/project/` | shared project directories |
| `/ceph/course/` | course material |
| `/ceph/container/` | ready-made container images |

File transfer with `scp` (Windows users may prefer WinSCP):

```bash
scp -r my_project/ user@student.aau.dk@ailab-fe01.srv.aau.dk:~/
scp user@student.aau.dk@ailab-fe01.srv.aau.dk:~/myfile.txt .
```

Shared project directory, group-writable and setgid:

```bash
chgrp <semester-group> my_project
chmod 770 my_project
chmod g+s my_project
```

### Running jobs (Slurm)

`srun` for short interactive work (< 1 hour: testing, debugging), `sbatch` for anything
long-running.

Key flags: `--mem=24G`, `--cpus-per-task=15`, `--gres=gpu:1`, `--time=04:00:00`.

Interactive one-liner inside a container:

```bash
srun --mem=24G --cpus-per-task=15 --gres=gpu:1 --time=01:00:00 \
  singularity exec --nv /ceph/container/python/python_3.10.sif \
  python3 -c "print('Hello from AI-LAB')"
```

Batch script (`sbatch my_job.sh`):

```bash
#!/bin/bash
#SBATCH --job-name=my_python_job
#SBATCH --output=my_job.out
#SBATCH --error=my_job.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

singularity exec --nv /ceph/container/python/python_3.10.sif python3 my_script.py
```

`--nv` is required for GPU access inside the container. Use `--begin` to schedule jobs
into low-demand windows (nights, weekends, holidays).

### Containers (Singularity)

Prefer the pre-pulled images, referenced by absolute path:

```bash
ls /ceph/container
# e.g. /ceph/container/pytorch/pytorch_24.09.sif
```

Pulling your own (set the cache dirs first, and pull *through* Slurm — not on the
front-end):

```bash
export SINGULARITY_TMPDIR="$HOME/.singularity/tmp/"
export SINGULARITY_CACHEDIR="$HOME/.singularity/cache/"
mkdir -p $SINGULARITY_CACHEDIR $SINGULARITY_TMPDIR
srun --mem 40G singularity pull docker://nvcr.io/nvidia/tensorflow:24.03-tf2-py3
```

Sources: NVIDIA NGC Catalog and Docker Hub. Building from a definition file uses
`singularity build --fakeroot $output_sif $input_def`, submitted with `sbatch`. Verify:

```bash
srun --gres=gpu:1 singularity exec --nv torch.sif python3 -c "import torch"
```

### Monitoring

```bash
squeue --me                      # my jobs
squeue                           # everything queued/running
sinfo                            # partition and node availability
scontrol show node <nodename>    # detail for one node
nodesummary                      # visual overview of all nodes
nvidia-smi                       # GPU utilisation
python3 /ceph/course/claaudia/docs/gpu_util.py
```

Healthy GPU utilisation during compute-heavy work is ~70–100%; temperature should stay
below 80 °C. Low utilisation means the pipeline is starving the GPU — fix the data
loading rather than asking for more GPUs.

### Limits and fair usage

- Max **8 running jobs** per user.
- Max **8 GPUs** per user across all jobs.
- Max **4 GPUs per job** (`--gres=gpu:4` / `-G 4`).
- Time limit: **default 1 hour, maximum 12 hours** — long training must checkpoint and
  requeue (see the checkpointing and requeuing guides).
- Holding GPUs open for interactive poking is **not allowed**; submit batch jobs.
- CPU-only workloads belong on UCloud, not AI-LAB.
- Educational and research use only. CLAAUDIA may kill jobs that disrupt the cluster.

### Data classification — important

Only **non-confidential data (AAU classification level 1)** may be used on AI-LAB.
AI-LAB is **not a storage platform**; treat `/ceph` as scratch for active work and keep
the authoritative copy elsewhere. Never place personal, sensitive or confidential data
in this pipeline's inputs or outputs on the cluster.

### Further guides

- System overview: https://hpc.aau.dk/ai-lab/system-overview/
- Fair usage: https://hpc.aau.dk/ai-lab/fair-usage/
- Checkpointing: https://hpc.aau.dk/ai-lab/guides/checkpointing/
- Requeuing jobs: https://hpc.aau.dk/ai-lab/guides/requeuing-jobs/
- Batch LLM inference: https://hpc.aau.dk/ai-lab/guides/batch-llm-inference/
- CI/CD with GitHub Actions: https://hpc.aau.dk/ai-lab/guides/ci-cd-with-github-actions/
- Service windows: https://hpc.aau.dk/ai-lab/service-windows/
