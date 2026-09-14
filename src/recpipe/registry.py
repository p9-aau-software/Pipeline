"""Name -> factory lookup, shared by dataset ingests and models.

The point of this module is that adding a model is a *new file*, never an edit to the
pipeline. Drop `src/recpipe/models/my_model.py` with an ``@MODELS.register("my_model")``
decorator and it becomes selectable as ``model=my_model`` on the command line.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable

# Modules that could not be imported during autoload, and why. Kept so that asking for a
# model whose optional dependency is missing produces the real reason instead of a bare
# "unknown model".
FAILED_IMPORTS: dict[str, str] = {}


class Registry:
    """A named collection of factories."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._entries:
                raise ValueError(f"{self._kind} {name!r} is already registered")
            self._entries[name] = factory
            return factory

        return decorator

    def build(self, name: str, **kwargs: Any) -> Any:
        if name not in self._entries:
            message = (
                f"unknown {self._kind} {name!r}. "
                f"Available: {', '.join(self.available()) or '(none)'}"
            )
            if FAILED_IMPORTS:
                broken = "; ".join(f"{mod}: {err}" for mod, err in FAILED_IMPORTS.items())
                message += f". Some modules failed to import and may define it -- {broken}"
            raise KeyError(message)
        return self._entries[name](**kwargs)

    def available(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries


def autoload(package_name: str, skip: tuple[str, ...] = ()) -> None:
    """Import every submodule of a package so its register decorators run.

    A module whose optional dependency is missing (torch outside the container, an
    external repo that is not checked out) is skipped rather than taking the whole
    registry down with it.
    """
    package = importlib.import_module(package_name)
    for module in pkgutil.iter_modules(package.__path__):
        if module.name.startswith("_") or module.name in skip:
            continue
        target = f"{package_name}.{module.name}"
        try:
            importlib.import_module(target)
        except ImportError as exc:
            FAILED_IMPORTS[target] = str(exc)


MODELS = Registry("model")
INGESTS = Registry("dataset ingest")
