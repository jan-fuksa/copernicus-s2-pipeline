from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple, Optional, Sequence


@dataclass(frozen=True)
class FileItem:
    role: str
    path: str
    band: Optional[str] = None


@dataclass(frozen=True)
class LevelFiles:
    root_dir: str
    items: list[FileItem]


@dataclass(frozen=True)
class ProductInfo:
    product_id: str
    product_name: str
    baseline: str = ""
    rel_orbit: str = ""
    cloud_cover: Optional[float] = None
    geofootprint: Optional[dict[str, Any]] = None
    coverage_ratio: Optional[float] = None


@dataclass(frozen=True)
class PairKey:
    tile_id: str
    sensing_start_utc: str  # ISO8601 "Z"


@dataclass(frozen=True)
class PairEntry:
    key: PairKey
    l1c: ProductInfo
    l2a: ProductInfo
    files_l1c: LevelFiles
    files_l2a: LevelFiles


@dataclass(frozen=True)
class DownloadManifest:
    manifest_version: str
    created_utc: str
    pipeline: dict[str, Any]
    query: dict[str, Any]
    output: dict[str, Any]
    pairs: list[PairEntry]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_manifest_json(manifest: DownloadManifest, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    def conv(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        return obj

    payload = json.loads(json.dumps(manifest, default=conv))
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def new_manifest(
    *,
    manifest_version: str,
    query: dict[str, Any],
    out_dir: str,
    layout: str,
    pairs: list[PairEntry],
) -> DownloadManifest:
    return DownloadManifest(
        manifest_version=manifest_version,
        created_utc=_utc_now_iso(),
        pipeline={"name": "s2pipe", "stage": "download", "stage_version": "1.0"},
        query=query,
        output={"out_dir": out_dir, "layout": layout},
        pairs=pairs,
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_id_now() -> str:
    # Folder-friendly run id
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _pair_key(pair_obj: dict[str, Any]) -> Tuple[str, str]:
    k = pair_obj.get("key") or {}
    return (str(k.get("tile_id", "")), str(k.get("sensing_start_utc", "")))


def merge_index_pairs(
    existing_pairs: list[dict[str, Any]],
    new_pairs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Keep-first merge, dedupe by (tile_id, sensing_start_utc). Returns (merged, added_count)."""
    merged = list(existing_pairs)
    seen = {_pair_key(p) for p in existing_pairs}
    added = 0
    for p in new_pairs:
        key = _pair_key(p)
        if key in seen:
            continue
        merged.append(p)
        seen.add(key)
        added += 1
    return merged, added


def update_download_index(
    *,
    index_path: Path,
    manifest_version: str,
    out_dir: str,
    layout: str,
    new_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Load (or create) an aggregated index.json and append new pairs (deduped)."""
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
    else:
        idx = {
            "manifest_version": manifest_version,
            "created_utc": utc_now_iso(),
            "pipeline": {"name": "s2pipe", "stage": "download", "stage_version": "1.0"},
            "query": {"aggregate": True},
            "output": {"out_dir": out_dir, "layout": layout},
            "pairs": [],
        }

    existing_pairs = idx.get("pairs") or []
    if not isinstance(existing_pairs, list):
        existing_pairs = []

    merged, _added = merge_index_pairs(existing_pairs, new_pairs)

    idx["manifest_version"] = manifest_version
    idx["created_utc"] = utc_now_iso()
    idx["pipeline"] = {"name": "s2pipe", "stage": "download", "stage_version": "1.0"}
    idx["query"] = {"aggregate": True}
    idx["output"] = {"out_dir": out_dir, "layout": layout}
    idx["pairs"] = merged

    atomic_write_json(index_path, idx)
    return idx
