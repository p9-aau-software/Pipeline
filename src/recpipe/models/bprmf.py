"""Matrix factorisation trained with the BPR pairwise ranking loss.

This is the reference implementation of the native integration level: it shows what a
model owns (its parameters, its loss, its checkpoint) and what the pipeline owns (data,
splits, sampling, metrics).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recpipe.data.dataset import BPRSampler, InteractionMatrix
from recpipe.models.base import RecModel, TrainContext, exclusion_lists, top_k
from recpipe.registry import MODELS
from recpipe.utils.runtime import resolve_device

CHECKPOINT_NAME = "checkpoint.pt"


class _MFModule(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int, init_std: float) -> None:
        super().__init__()
        self.user = nn.Parameter(torch.empty(n_users, dim).normal_(0.0, init_std))
        self.item = nn.Parameter(torch.empty(n_items, dim).normal_(0.0, init_std))
        self.item_bias = nn.Parameter(torch.zeros(n_items))

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user[users] * self.item[items]).sum(-1) + self.item_bias[items]

    def scores_for(self, users: torch.Tensor) -> torch.Tensor:
        return self.user[users] @ self.item.T + self.item_bias


@MODELS.register("bprmf")
class BPRMF(RecModel):
    def __init__(
        self,
        embedding_dim: int = 64,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-5,
        batch_size: int = 1024,
        init_std: float = 0.1,
        **_: object,
    ) -> None:
        self.embedding_dim = int(embedding_dim)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.init_std = float(init_std)

        self.module: _MFModule | None = None
        self.device = "cpu"
        self.history: list[dict[str, float]] = []

    def fit(self, train: InteractionMatrix, ctx: TrainContext) -> None:
        self.device = resolve_device(ctx.device)
        self.module = _MFModule(
            train.n_users, train.n_items, self.embedding_dim, self.init_std
        ).to(self.device)

        optimiser = torch.optim.Adam(
            self.module.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        sampler = BPRSampler(train, seed=ctx.seed)

        # Resume if a previous run of this job was interrupted (Slurm requeue).
        start_epoch = 0
        checkpoint = (ctx.run_dir / CHECKPOINT_NAME) if ctx.run_dir else None
        if checkpoint is not None and checkpoint.exists():
            start_epoch = self._restore(checkpoint, optimiser)
            print(f"resuming from epoch {start_epoch}")

        for epoch in range(start_epoch, ctx.epochs):
            total, seen = 0.0, 0
            for users, positive, negative in sampler.batches(self.batch_size):
                u = torch.as_tensor(users, device=self.device)
                i = torch.as_tensor(positive, device=self.device)
                j = torch.as_tensor(negative, device=self.device)

                loss = -F.logsigmoid(self.module.score(u, i) - self.module.score(u, j)).mean()

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                total += float(loss) * len(users)
                seen += len(users)

            mean_loss = total / max(seen, 1)
            self.history.append({"epoch": epoch + 1, "loss": mean_loss})
            if ctx.log_every and (epoch + 1) % ctx.log_every == 0:
                print(f"epoch {epoch + 1}/{ctx.epochs}  loss {mean_loss:.4f}")

            if checkpoint is not None and ctx.checkpoint_every:
                if (epoch + 1) % ctx.checkpoint_every == 0:
                    self._store(checkpoint, optimiser, epoch + 1)

    @torch.no_grad()
    def recommend(
        self, users: np.ndarray, k: int, exclude: InteractionMatrix | None = None
    ) -> np.ndarray:
        if self.module is None:
            raise RuntimeError("BPRMF.recommend called before fit")
        tensor = torch.as_tensor(np.asarray(users), device=self.device)
        scores = self.module.scores_for(tensor).cpu().numpy()
        return top_k(scores, k, exclusion_lists(users, exclude))

    def _store(self, path: Path, optimiser: torch.optim.Optimizer, epoch: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.module.state_dict(),
                "optimiser": optimiser.state_dict(),
                "epoch": epoch,
                "history": self.history,
            },
            path,
        )

    def _restore(self, path: Path, optimiser: torch.optim.Optimizer) -> int:
        payload = torch.load(path, map_location=self.device)
        self.module.load_state_dict(payload["model"])
        optimiser.load_state_dict(payload["optimiser"])
        self.history = payload.get("history", [])
        return int(payload["epoch"])

    def save(self, path: Path) -> None:
        if self.module is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.module.state_dict(), "history": self.history}, path)
