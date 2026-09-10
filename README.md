# Pipeline

A recommender-system experiment pipeline for AAU AI-LAB. Swapping the dataset, the model
or the job size is a config change; the pipeline itself stays put.

```bash
python -m recpipe.train model=bprmf data=ml100k trainer=default
python scripts/submit.py model=bprmf data=ml100k slurm=default
python scripts/compare.py
```

## Setup on AI-LAB

```bash
git clone https://github.com/p9-aau-software/Pipeline.git
cd Pipeline
bash scripts/setup_env.sh
```

That builds a venv on `/ceph` inside the shared PyTorch container, so `torch` and CUDA
come from the container and our own packages sit on top. Override the container or venv
location with `RECPIPE_CONTAINER` / `RECPIPE_VENV` (see `scripts/env.sh`).

Datasets are downloaded on demand, which needs internet — build the cache on a front-end
node before submitting jobs:

```bash
python -m recpipe.data.prepare data=ml100k
```

## Running things

| Command | What it does |
|---|---|
| `python -m recpipe.train` | train + evaluate one config (defaults: synthetic + popularity) |
| `python scripts/submit.py <overrides>` | same, as a Slurm job with a resource profile |
| `python -m recpipe.export_split` | write the split for an outside repo to train on |
| `python -m recpipe.evaluate predictions=<file>` | score predictions produced elsewhere |
| `python scripts/compare.py` | all runs in one leaderboard |
| `pytest` | fast CPU checks on synthetic data |

Any config value can be overridden on the command line:

```bash
python -m recpipe.train model=bprmf model.params.embedding_dim=128 trainer.epochs=50
python scripts/submit.py model=bprmf slurm=default --dry-run
```

Run `pytest` before submitting anything. It exercises the whole chain on a 200-user
synthetic dataset in seconds, which is a cheaper place to find a bug than a GPU queue.

## Layout

```
conf/           Hydra configs: data/ model/ task/ trainer/ slurm/
src/recpipe/
  data/         ingest -> canonical parquet -> filter/remap -> split -> sample
  models/       registry, the shared interface, and the models themselves
  eval/         metrics and the evaluator both model and prediction paths go through
  train.py      the entrypoint
scripts/        environment setup, job submission, result comparison
outputs/        one directory per run: config, metrics, checkpoints (gitignored)
```

## Adding a dataset

Write an ingest that returns the canonical columns (`user_id`, `item_id`, `timestamp`,
`weight`) — see `src/recpipe/data/ingest/movielens.py`:

```python
@INGESTS.register("my_dataset")
def my_dataset(raw_dir="data/raw", **_):
    frame = pd.read_csv(Path(raw_dir) / "my_data.csv")
    frame = frame.rename(columns={"customer": schema.USER, "product": schema.ITEM})
    frame[schema.WEIGHT] = 1.0
    return schema.validate(frame, source="my_dataset")
```

Then add `conf/data/my_dataset.yaml` and run `data=my_dataset`. Nothing downstream
changes: filtering, id remapping, caching, splits, sampling and evaluation are shared.

## Adding a model

Drop a file in `src/recpipe/models/` with the decorator and the two methods:

```python
@MODELS.register("my_model")
class MyModel(RecModel):
    def fit(self, train: InteractionMatrix, ctx: TrainContext) -> None: ...
    def recommend(self, users, k, exclude=None) -> np.ndarray: ...
```

Add `conf/model/my_model.yaml` and run `model=my_model`. The registry finds the file on
its own — no import to register anywhere.

For models from other people's repositories, see
[`src/recpipe/models/external/README.md`](src/recpipe/models/external/README.md).

## Cluster notes

`scripts/submit.py` refuses jobs above AI-LAB's limits (4 GPUs per job, 12 hours) before
they reach the queue. The other limits — 8 jobs and 8 GPUs per user — are yours to watch
with `squeue --me`. Cluster specifics live in [CLAUDE.md](CLAUDE.md).
