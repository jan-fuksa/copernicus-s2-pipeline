from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunningStats:
    """Streaming mean/std per channel."""

    count: int
    mean: Any
    m2: Any


def stats_init(num_channels: int) -> RunningStats:
    raise NotImplementedError


def stats_update(
    stats: RunningStats, x: Any, max_pixels: int | None = None
) -> RunningStats:
    """Update running stats from one sample tensor x (C,H,W)."""
    raise NotImplementedError


def stats_finalize(stats: RunningStats) -> dict[str, Any]:
    """Return serializable dict with mean/std."""
    raise NotImplementedError


def stats_save(stats_obj: dict[str, Any], path: Path) -> None:
    raise NotImplementedError


def stats_load(path: Path) -> dict[str, Any]:
    raise NotImplementedError


def apply_normalization(x: Any, stats_obj: dict[str, Any]) -> Any:
    raise NotImplementedError
