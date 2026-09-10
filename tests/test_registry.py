"""Adding a model should be a new file, and asking for a missing one should say so."""

from __future__ import annotations

import pytest

from recpipe import models as _models  # noqa: F401  (registers the models)
from recpipe.data import ingest as _ingest  # noqa: F401  (registers the ingests)
from recpipe.registry import INGESTS, MODELS, Registry


def test_builtin_models_are_discovered():
    assert "popularity" in MODELS.available()


def test_builtin_ingests_are_discovered():
    assert {"synthetic", "movielens"} <= set(INGESTS.available())


def test_unknown_name_lists_what_is_available():
    with pytest.raises(KeyError) as excinfo:
        MODELS.build("no_such_model")
    assert "popularity" in str(excinfo.value)


def test_double_registration_is_rejected():
    registry = Registry("thing")
    registry.register("a")(lambda: 1)
    with pytest.raises(ValueError):
        registry.register("a")(lambda: 2)


def test_registered_factory_receives_parameters():
    registry = Registry("thing")

    @registry.register("configurable")
    class Thing:
        def __init__(self, size: int = 1) -> None:
            self.size = size

    assert registry.build("configurable", size=7).size == 7
