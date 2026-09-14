"""MovieLens ingest -- the reference for what a real ingest looks like.

Downloading needs internet, which compute nodes may not have. Build the cache on a
front-end node first:

    python -m recpipe.data.prepare data=ml100k
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from recpipe.data import schema
from recpipe.registry import INGESTS

_VARIANTS = {
    "ml-100k": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "member": "ml-100k/u.data",
        "sep": "\t",
    },
    "ml-1m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "member": "ml-1m/ratings.dat",
        "sep": "::",
    },
}


def _download(variant: str, raw_dir: Path) -> Path:
    spec = _VARIANTS[variant]
    archive = raw_dir / f"{variant}.zip"
    extracted = raw_dir / spec["member"]

    if extracted.exists():
        return extracted

    raw_dir.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        print(f"downloading {spec['url']} -> {archive}")
        urllib.request.urlretrieve(spec["url"], archive)

    with zipfile.ZipFile(archive) as zf:
        zf.extract(spec["member"], path=raw_dir)
    return extracted


@INGESTS.register("movielens")
def movielens(
    raw_dir: str = "data/raw",
    variant: str = "ml-100k",
    min_rating: float = 4.0,
    **_: object,
) -> pd.DataFrame:
    if variant not in _VARIANTS:
        raise ValueError(f"unknown MovieLens variant {variant!r}; pick one of {list(_VARIANTS)}")

    path = _download(variant, Path(raw_dir))
    spec = _VARIANTS[variant]

    frame = pd.read_csv(
        path,
        sep=spec["sep"],
        engine="python",
        names=[schema.USER, schema.ITEM, "rating", schema.TIMESTAMP],
        encoding="latin-1",
    )

    # Implicit feedback: a rating at or above the threshold counts as a positive signal,
    # everything else is dropped rather than treated as a negative.
    if min_rating is not None:
        frame = frame[frame["rating"] >= min_rating]

    frame[schema.WEIGHT] = 1.0
    return schema.validate(frame, source=f"movielens ingest ({variant})")
