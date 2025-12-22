from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence


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
