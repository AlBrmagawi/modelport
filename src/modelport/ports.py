"""Narrow boundaries implemented by filesystem, SQLite and CPU adapters."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from modelport.config import Settings
from modelport.domain import ModelDescriptor


class Inspector(Protocol):
    def inspect(self, directory: Path) -> ModelDescriptor: ...


class Runtime(Protocol):
    info: dict[str, Any]

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]: ...
    def close(self) -> None: ...


class InputProvider(Protocol):
    identity: dict[str, Any]

    def batches(self) -> Iterator[dict[str, Any]]: ...


@dataclass
class ExecutionContext:
    work_dir: Path
    settings: Settings
    progress: Callable[[str, dict[str, Any]], None]
    check_deadline: Callable[[], None]

    def stage(self, name: str, **details: Any) -> None:
        self.check_deadline()
        self.progress(name, details)
