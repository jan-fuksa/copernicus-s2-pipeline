from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    raw_l1c: Path
    raw_l2a: Path
    manifest_dir: Path
    tmp: Path


def make_paths(out_dir: Path) -> OutputPaths:
    root = out_dir
    raw_l1c = root / "raw" / "L1C"
    raw_l2a = root / "raw" / "L2A"
    manifest_dir = root / "meta" / "manifest"
    tmp = root / "tmp"
    return OutputPaths(root=root, raw_l1c=raw_l1c, raw_l2a=raw_l2a, manifest_dir=manifest_dir, tmp=tmp)


def ensure_dirs(paths: OutputPaths) -> None:
    for p in (paths.root, paths.raw_l1c, paths.raw_l2a, paths.manifest_dir, paths.tmp):
        p.mkdir(parents=True, exist_ok=True)
