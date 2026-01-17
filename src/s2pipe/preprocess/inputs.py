"""
inputs.py
---------
Utilities for Step-2 preprocessing that read the Step-1 download manifest (index.json)
and resolve/select the concrete on-disk assets needed for a given scene.

Responsibilities:
- Parse index.json into typed dataclasses (DownloadIndex, IndexScene, ProductMeta, IndexFiles, ...).
- Resolve relative paths stored in the manifest to absolute paths under out_dir.
- Select required assets based on a set of required roles with presence validation,
  returning a single SelectedAssets object suitable for downstream preprocessing steps.

Notes:
- select_assets() is the preferred unified selector API.

--------------------------------------------------------------------
Suggested "public API" imports (recommended)

Core functions
- load_download_index(index_path):
    Parse Step-1 index.json into a DownloadIndex typed structure.
- iter_scenes(index, require_present=False):
    Iterate over IndexScene items; optional strict presence check for all indexed files.
- resolve_path(index, rel_path):
    Convert a manifest path (relative to out_dir) to an absolute Path.
- select_assets(scene, index, ...):
    Unified asset selector; returns SelectedAssets with resolved absolute paths and
    passthrough metadata.

Core dataclasses / types
- DownloadIndex:
    Root object: out_dir + list of IndexScene items.
- IndexScene:
    One (tile_id, sensing_start_utc) scene linking L1C/L2A metas and file inventories.
- SelectedAssets:
    Output of selection (absolute paths for bands/metadata/SCL + metadata passthrough).
- ProductMeta:
    Product-level metadata (IDs, orbit, cloud cover, coverage ratio, etc.).
- IndexFiles, IndexFileItem:
    File inventory for a product (role/path/planned/present).


This module intentionally does not provide any legacy selector wrappers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


STEP1_SCHEMA_V2 = "s2pipe.step1.index.v2"


@dataclass(frozen=True)
class IndexFileItem:
    # Example roles:
    # - "l1c.band.B02"
    # - "l1c.product_metadata"
    # - "l2a.scl_20m"
    role: str
    path: str  # relative to out_dir
    planned: bool
    present: bool


@dataclass(frozen=True)
class IndexFiles:
    root_dir: str  # e.g. "raw/L1C/.../files"
    items: list[IndexFileItem]


@dataclass(frozen=True)
class ProductMeta:
    product_id: str
    product_name: str
    baseline: str | None
    rel_orbit: str | None
    cloud_cover: float | None
    coverage_ratio: float | None
    geofootprint: dict[str, Any] | None
    scl_percentages: dict[str, float] | None


@dataclass(frozen=True)
class IndexScene:
    tile_id: str
    sensing_start_utc: str
    l1c: ProductMeta
    l2a: ProductMeta
    files_l1c: IndexFiles
    files_l2a: IndexFiles


@dataclass(frozen=True)
class DownloadIndex:
    out_dir: Path
    scenes: list[IndexScene]


@dataclass(frozen=True)
class SelectedAssets:
    """Resolved absolute paths for assets selected from the Step-1 index."""

    paths_by_role: dict[str, Path]

    # Passthrough metadata (may be used in Step-2 exports).
    cloud_cover: float | None
    coverage_ratio: float | None
    scl_percentages: dict[str, float] | None

    def get(self, role: str) -> Path | None:
        return self.paths_by_role.get(role)

    def require(self, role: str) -> Path:
        p = self.get(role)
        if p is None:
            raise FileNotFoundError(f"Missing required asset role={role!r}")
        return p

    def get_l1c_band(self, band: str) -> Path | None:
        return self.get(f"l1c.band.{band}")

    def require_l1c_band(self, band: str) -> Path:
        return self.require(f"l1c.band.{band}")

    def subset(self, roles: Iterable[str]) -> dict[str, Path]:
        """Return a subset of paths_by_role containing the given roles."""
        out: dict[str, Path] = {}
        for r in roles:
            p = self.get(r)
            if p is not None:
                out[r] = p
        return out


def _as_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y"}
    return default


def _as_float_or_none(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _as_str_or_none(x: Any) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def _resolve_out_dir(index_path: Path, out_dir_str: str | None) -> Path:
    """Resolve dataset root directory.

    Step-1 index.json is typically at:
      <out_dir>/meta/step1/index.json

    If out_dir_str is absolute, trust it.
    If it is relative (or missing), prefer the directory implied by index_path location.
    """
    if out_dir_str:
        p = Path(out_dir_str)
        if p.is_absolute():
            return p

    # Candidate derived from index.json location: <out_dir>/meta/step1/index.json
    try:
        candidate = index_path.parents[2]
        if (candidate / "meta").exists() or (candidate / "raw").exists():
            return candidate
    except Exception:
        pass

    if out_dir_str:
        return (Path.cwd() / out_dir_str).resolve()

    return index_path.parent.resolve()


def _parse_product_meta(d: dict[str, Any], *, context: str) -> ProductMeta:
    if not isinstance(d, dict):
        raise ValueError(f"{context}: expected dict, got {type(d)}")

    pid = d.get("product_id")
    pname = d.get("product_name")
    if not isinstance(pid, str) or not pid:
        raise ValueError(f"{context}: missing/invalid product_id")
    if not isinstance(pname, str) or not pname:
        raise ValueError(f"{context}: missing/invalid product_name")

    return ProductMeta(
        product_id=pid,
        product_name=pname,
        baseline=_as_str_or_none(d.get("baseline")),
        rel_orbit=_as_str_or_none(d.get("rel_orbit")),
        cloud_cover=_as_float_or_none(d.get("cloud_cover")),
        coverage_ratio=_as_float_or_none(d.get("coverage_ratio")),
        geofootprint=d.get("geofootprint")
        if isinstance(d.get("geofootprint"), dict)
        else None,
        scl_percentages=d.get("scl_percentages")
        if isinstance(d.get("scl_percentages"), dict)
        else None,
    )


def _parse_files(d: dict[str, Any], *, context: str) -> IndexFiles:
    if not isinstance(d, dict):
        raise ValueError(f"{context}: expected dict, got {type(d)}")

    root_dir = d.get("root_dir")
    if not isinstance(root_dir, str) or not root_dir:
        raise ValueError(f"{context}: missing/invalid root_dir")

    items_raw = d.get("items")
    if not isinstance(items_raw, list):
        raise ValueError(f"{context}: missing/invalid items")

    items: list[IndexFileItem] = []
    for i, it in enumerate(items_raw):
        if not isinstance(it, dict):
            raise ValueError(f"{context}: items[{i}] expected dict, got {type(it)}")

        role = it.get("role")
        path = it.get("path")

        if not isinstance(role, str) or not role:
            raise ValueError(f"{context}: items[{i}] missing/invalid role")
        if not isinstance(path, str) or not path:
            raise ValueError(f"{context}: items[{i}] missing/invalid path")

        items.append(
            IndexFileItem(
                role=role,
                path=path,
                planned=_as_bool(it.get("planned"), default=False),
                present=_as_bool(it.get("present"), default=False),
            )
        )

    return IndexFiles(root_dir=root_dir, items=items)


def load_download_index(index_path: Path) -> DownloadIndex:
    """Parse Step-1 index.json into typed structures."""
    index_path = Path(index_path)
    js = json.loads(index_path.read_text(encoding="utf-8"))

    if not isinstance(js, dict):
        raise ValueError(f"index.json: expected dict, got {type(js)}")

    schema = js.get("schema")
    if schema != STEP1_SCHEMA_V2:
        raise ValueError(
            f"index.json: unsupported schema={schema!r}; expected {STEP1_SCHEMA_V2!r}"
        )

    out = js.get("output") if isinstance(js.get("output"), dict) else {}
    out_dir_str = out.get("out_dir") if isinstance(out, dict) else None
    out_dir = _resolve_out_dir(index_path, out_dir_str)

    scenes_raw = js.get("scenes")
    if not isinstance(scenes_raw, list):
        raise ValueError("index.json: missing/invalid 'scenes' list")

    scenes: list[IndexScene] = []
    for i, p in enumerate(scenes_raw):
        if not isinstance(p, dict):
            raise ValueError(f"scenes[{i}]: expected dict, got {type(p)}")

        key = p.get("key")
        if not isinstance(key, dict):
            raise ValueError(f"scenes[{i}]: missing/invalid key")

        tile_id = key.get("tile_id")
        sensing = key.get("sensing_start_utc")
        if not isinstance(tile_id, str) or not tile_id:
            raise ValueError(f"scenes[{i}]: missing/invalid key.tile_id")
        if not isinstance(sensing, str) or not sensing:
            raise ValueError(f"scenes[{i}]: missing/invalid key.sensing_start_utc")

        l1c = _parse_product_meta(p.get("l1c", {}), context=f"scenes[{i}].l1c")
        l2a = _parse_product_meta(p.get("l2a", {}), context=f"scenes[{i}].l2a")
        files_l1c = _parse_files(
            p.get("files_l1c", {}), context=f"scenes[{i}].files_l1c"
        )
        files_l2a = _parse_files(
            p.get("files_l2a", {}), context=f"scenes[{i}].files_l2a"
        )

        scenes.append(
            IndexScene(
                tile_id=tile_id,
                sensing_start_utc=sensing,
                l1c=l1c,
                l2a=l2a,
                files_l1c=files_l1c,
                files_l2a=files_l2a,
            )
        )

    return DownloadIndex(out_dir=out_dir, scenes=scenes)


def iter_scenes(
    index: DownloadIndex, *, require_present: bool = False
) -> Iterator[IndexScene]:
    """Yield scenes from the index.

    Note:
      - Step 2 usually validates presence *after* asset selection (bands/SCL/metadata).
      - If require_present=True, this function applies a strict rule: ALL indexed items must be present.
        This may be overly strict if your index contains more assets than you plan to use.
    """
    for p in index.scenes:
        if not require_present:
            yield p
            continue

        all_present = True
        for it in p.files_l1c.items + p.files_l2a.items:
            if not it.present:
                all_present = False
                break
        if all_present:
            yield p


def resolve_path(index: DownloadIndex, rel_path: str) -> Path:
    """Resolve a path stored in index.json (relative to out_dir) to an absolute Path."""
    return (index.out_dir / Path(rel_path)).resolve()


def select_assets(
    scene: IndexScene,
    index: DownloadIndex,
    *,
    required_roles: set[str],
    require_present: bool = True,
) -> SelectedAssets:
    """Select and resolve assets from one index scene.

    The caller provides a set of required roles (e.g. {"l1c.band.B02", "l1c.product_metadata"}).
    This function validates that each role exists in the scene inventory and, optionally,
    that the corresponding file is present on disk.
    """

    items = scene.files_l1c.items + scene.files_l2a.items

    by_role: dict[str, IndexFileItem] = {}
    dup: set[str] = set()
    for it in items:
        if it.role in by_role:
            dup.add(it.role)
            continue
        by_role[it.role] = it

    if dup:
        raise ValueError(
            f"Duplicate roles in scene inventory for tile={scene.tile_id} sensing={scene.sensing_start_utc}: {sorted(dup)}"
        )

    missing: list[str] = [r for r in sorted(required_roles) if r not in by_role]
    if missing:
        raise FileNotFoundError(
            f"Missing required roles for tile={scene.tile_id} sensing={scene.sensing_start_utc}: {missing}"
        )

    not_present: list[str] = [
        r for r in sorted(required_roles) if require_present and not by_role[r].present
    ]
    if not_present:
        raise FileNotFoundError(
            f"Required roles exist but are not present on disk for tile={scene.tile_id} sensing={scene.sensing_start_utc}: {not_present}"
        )

    paths_by_role = {r: resolve_path(index, by_role[r].path) for r in required_roles}

    return SelectedAssets(
        paths_by_role=paths_by_role,
        cloud_cover=scene.l1c.cloud_cover,
        coverage_ratio=scene.l1c.coverage_ratio,
        scl_percentages=scene.l2a.scl_percentages,
    )
