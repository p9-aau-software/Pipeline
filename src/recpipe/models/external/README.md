# Running someone else's model

Foreign recommender repos rarely fit a shared interface. They bring their own training
loop, their own data format and their own config, and rewriting each one is exactly the
work this pipeline exists to avoid. Pick the cheapest level that works.

Whatever you do, the numbers must come out of **our** evaluator. A result computed by
someone else's evaluation code is not comparable to a result from this repo, even when
both call the metric "NDCG@10" — the split, the exclusion of seen items and the
tie-breaking all differ.

## Level 2 — the repo gives you a scorer

Use this when you can import their model and get scores out of it. You keep our data,
our splits, our sampling and our metrics; only the parameters and the forward pass are
theirs.

1. Vendor the code: `git clone <repo> src/recpipe/models/external/<name>`
   (the `.gitignore` keeps their source out of our history — add it as a submodule
   instead if you want it pinned).
2. Write an adapter next to it:

```python
# src/recpipe/models/their_model.py
import numpy as np

from recpipe.data.dataset import BPRSampler, InteractionMatrix
from recpipe.models.base import RecModel, TrainContext, exclusion_lists, top_k
from recpipe.registry import MODELS


@MODELS.register("their_model")
class TheirModel(RecModel):
    def __init__(self, embedding_dim: int = 64, **_):
        self.embedding_dim = embedding_dim
        self.net = None

    def fit(self, train: InteractionMatrix, ctx: TrainContext) -> None:
        from recpipe.models.external.their_repo.model import TheirNet  # their code

        self.net = TheirNet(train.n_users, train.n_items, self.embedding_dim)
        # ... our training loop, or theirs if it takes an arbitrary iterator

    def recommend(self, users, k, exclude=None):
        scores = self.net.score_all(users)          # (len(users), n_items)
        return top_k(scores, k, exclusion_lists(users, exclude))
```

3. Add `conf/model/their_model.yaml` and run it like any other model.

If their code needs incompatible dependencies, give it its own venv rather than
disturbing the main one:

```bash
RECPIPE_VENV=$HOME/venvs/their_model bash scripts/setup_env.sh
```

## Level 3 — the repo is not worth touching

Use this when their code only runs as a whole: its own CLI, its own data loader, its own
checkpoints. Run it unchanged and score its output.

```bash
# 1. hand them exactly our split
python -m recpipe.export_split data=ml100k export_dir=exports/ml100k

# 2. run their code however its README says, pointed at exports/ml100k
#    (train.csv / val.csv / test.csv, plus parquet if they read it)

# 3. score whatever they produced
python -m recpipe.evaluate data=ml100k predictions=exports/ml100k/predictions.parquet
```

`predictions.parquet` (or `.csv`) needs:

| column    | meaning                                          |
|-----------|--------------------------------------------------|
| `user_id` | the exported index, **not** the original id      |
| `item_id` | the exported index                               |
| `rank`    | 0 is the top recommendation (or give `score`)    |

The ids are the usual thing that goes wrong. `exports/<name>/user_ids.npy` and
`item_ids.npy` map indices back to the dataset's own identifiers if their code insists
on the originals — convert on the way out, before evaluating.

The resulting run lands in `outputs/` alongside everything else, so
`python scripts/compare.py` puts it in the same table as your own models.
