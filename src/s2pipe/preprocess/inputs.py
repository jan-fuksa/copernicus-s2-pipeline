"""
inputs.py
---------
Utilities for Step-2 preprocessing that read the Step-1 download manifest (index.json)
and resolve/select the concrete on-disk assets needed for a given L1C/L2A pair.

Responsibilities:
- Parse index.json into typed dataclasses (DownloadIndex, IndexPair, ProductMeta, IndexFiles, ...).
- Resolve relative paths stored in the manifest to absolute paths under out_dir.
- Select required assets (L1C bands, tile metadata XML, SCL_20m) with presence validation,
  returning a single SelectedAssets object suitable for downstream preprocessing steps.

Notes:
- select_assets() is the preferred unified selector API.
- Backwards-compatible wrapper selectors are kept temporarily for legacy callers.

--------------------------------------------------------------------
Suggested "public API" imports (recommended)

Core functions
- load_download_index(index_path):
    Parse Step-1 index.json into a DownloadIndex typed structure.
- iter_pairs(index, require_present=False):
    Iterate over IndexPair items; optional strict presence check for all indexed files.
- resolve_path(index, rel_path):
    Convert a manifest path (relative to out_dir) to an absolute Path.
- select_assets(pair, index, ...):
    Unified asset selector; returns SelectedAssets with resolved absolute paths and
    passthrough metadata.

Core dataclasses / types
- DownloadIndex:
    Root object: out_dir + list of IndexPair items.
- IndexPair:
    One (tile_id, sensing_start_utc) pair linking L1C/L2A metas and file inventories.
- SelectedAssets:
    Output of selection (absolute paths for bands/metadata/SCL + metadata passthrough).
- ProductMeta:
    Product-level metadata (IDs, orbit, cloud cover, coverage ratio, etc.).
- IndexFiles, IndexFileItem:
    File inventory for a product (role/path/band/planned/present).

Legacy wrappers (optional; keep only if still used elsewhere)
- select_l1c_band_paths(...):
    Legacy convenience wrapper returning dict[band, Path].
- select_tile_metadata_path(..., level="L1C"|"L2A"):
    Legacy wrapper returning Path to MTD_TL.xml.
- select_scl_20m_path(...):
    Legacy wrapper returning Path to SCL 20m raster.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence


@dataclass(frozen=True)
class IndexFileItem:
    role: str  # "band" | "tile_metadata" | "scl_20m" | ...
    path: str  # relative to out_dir
    band: Optional[str]  # e.g. "B02" or None
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
class IndexPair:
    tile_id: str
    sensing_start_utc: str
    l1c: ProductMeta
    l2a: ProductMeta
    files_l1c: IndexFiles
    files_l2a: IndexFiles


@dataclass(frozen=True)
class DownloadIndex:
    out_dir: Path
    pairs: list[IndexPair]


@dataclass(frozen=True)
class SelectedAssets:
    """Resolved absolute paths for assets required by preprocessing."""

    # L1C
    l1c_bands: dict[str, Path]  # band -> absolute path
    l1c_tile_metadata: Path | None  # MTD_TL.xml (optional)

    # L2A
    l2a_tile_metadata: Path | None  # MTD_TL.xml (optional)
    scl_20m: Path | None  # SCL_20m.jp2 (optional)

    # passthrough metadata (useful for Step 2 manifest)
    cloud_cover: float | None
    coverage_ratio: float | None
    scl_percentages: dict[str, float] | None


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
        band = it.get("band")

        if not isinstance(role, str) or not role:
            raise ValueError(f"{context}: items[{i}] missing/invalid role")
        if not isinstance(path, str) or not path:
            raise ValueError(f"{context}: items[{i}] missing/invalid path")

        items.append(
            IndexFileItem(
                role=role,
                path=path,
                band=str(band) if isinstance(band, str) and band else None,
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

    out = js.get("output") if isinstance(js.get("output"), dict) else {}
    out_dir_str = out.get("out_dir") if isinstance(out, dict) else None
    out_dir = _resolve_out_dir(index_path, out_dir_str)

    pairs_raw = js.get("pairs")
    if not isinstance(pairs_raw, list):
        raise ValueError("index.json: missing/invalid 'pairs' list")

    pairs: list[IndexPair] = []
    for i, p in enumerate(pairs_raw):
        if not isinstance(p, dict):
            raise ValueError(f"pairs[{i}]: expected dict, got {type(p)}")

        key = p.get("key")
        if not isinstance(key, dict):
            raise ValueError(f"pairs[{i}]: missing/invalid key")

        tile_id = key.get("tile_id")
        sensing = key.get("sensing_start_utc")
        if not isinstance(tile_id, str) or not tile_id:
            raise ValueError(f"pairs[{i}]: missing/invalid key.tile_id")
        if not isinstance(sensing, str) or not sensing:
            raise ValueError(f"pairs[{i}]: missing/invalid key.sensing_start_utc")

        l1c = _parse_product_meta(p.get("l1c", {}), context=f"pairs[{i}].l1c")
        l2a = _parse_product_meta(p.get("l2a", {}), context=f"pairs[{i}].l2a")
        files_l1c = _parse_files(
            p.get("files_l1c", {}), context=f"pairs[{i}].files_l1c"
        )
        files_l2a = _parse_files(
            p.get("files_l2a", {}), context=f"pairs[{i}].files_l2a"
        )

        pairs.append(
            IndexPair(
                tile_id=tile_id,
                sensing_start_utc=sensing,
                l1c=l1c,
                l2a=l2a,
                files_l1c=files_l1c,
                files_l2a=files_l2a,
            )
        )

    return DownloadIndex(out_dir=out_dir, pairs=pairs)


def iter_pairs(
    index: DownloadIndex, *, require_present: bool = False
) -> Iterator[IndexPair]:
    """Yield pairs from the index.

    Note:
      - Step 2 usually validates presence *after* asset selection (bands/SCL/metadata).
      - If require_present=True, this function applies a strict rule: ALL indexed items must be present.
        This may be overly strict if your index contains more assets than you plan to use.
    """
    for p in index.pairs:
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


def _pick_single_item(
    items: list[IndexFileItem],
    *,
    role: str,
    band: str | None = None,
    require_present: bool,
) -> IndexFileItem:
    cand = [it for it in items if it.role == role and (band is None or it.band == band)]
    if not cand:
        raise FileNotFoundError(f"Missing item role={role!r} band={band!r}")

    cand_present = [it for it in cand if it.present]
    if cand_present:
        return cand_present[0]

    if require_present:
        raise FileNotFoundError(
            f"Item role={role!r} band={band!r} exists only as planned/non-present"
        )

    return cand[0]


def select_assets(
    pair: IndexPair,
    index: DownloadIndex,
    *,
    l1c_bands: Sequence[str],
    need_l1c_tile_metadata: bool = True,
    need_l2a_tile_metadata: bool = False,
    need_scl_20m: bool = True,
    require_present: bool = True,
) -> SelectedAssets:
    """Select and resolve required assets from one index pair."""
    # L1C bands
    bands_out: dict[str, Path] = {}
    missing_bands: list[str] = []
    for b in l1c_bands:
        try:
            it = _pick_single_item(
                pair.files_l1c.items,
                role="band",
                band=str(b),
                require_present=require_present,
            )
            bands_out[str(b)] = resolve_path(index, it.path)
        except FileNotFoundError:
            missing_bands.append(str(b))
    if missing_bands:
        raise FileNotFoundError(
            f"Missing required L1C bands for tile={pair.tile_id} sensing={pair.sensing_start_utc}: {missing_bands}"
        )

    # Tile metadata
    l1c_mtd: Path | None = None
    if need_l1c_tile_metadata:
        it = _pick_single_item(
            pair.files_l1c.items,
            role="tile_metadata",
            band=None,
            require_present=require_present,
        )
        l1c_mtd = resolve_path(index, it.path)

    l2a_mtd: Path | None = None
    if need_l2a_tile_metadata:
        it = _pick_single_item(
            pair.files_l2a.items,
            role="tile_metadata",
            band=None,
            require_present=require_present,
        )
        l2a_mtd = resolve_path(index, it.path)

    # SCL
    scl_path: Path | None = None
    if need_scl_20m:
        it = _pick_single_item(
            pair.files_l2a.items,
            role="scl_20m",
            band=None,
            require_present=require_present,
        )
        scl_path = resolve_path(index, it.path)

    return SelectedAssets(
        l1c_bands=bands_out,
        l1c_tile_metadata=l1c_mtd,
        l2a_tile_metadata=l2a_mtd,
        scl_20m=scl_path,
        cloud_cover=pair.l1c.cloud_cover,
        coverage_ratio=pair.l1c.coverage_ratio,
        scl_percentages=pair.l2a.scl_percentages,
    )


# ---------------------------------------------------------------------
# Backwards-compatible wrappers (optional).
# You may remove these once callers are migrated to select_assets().
# ---------------------------------------------------------------------


def select_l1c_band_paths(
    pair: IndexPair,
    index: DownloadIndex,
    *,
    bands: list[str],
    require_present: bool = True,
) -> dict[str, Path]:
    return select_assets(
        pair,
        index,
        l1c_bands=bands,
        need_l1c_tile_metadata=False,
        need_l2a_tile_metadata=False,
        need_scl_20m=False,
        require_present=require_present,
    ).l1c_bands


def select_tile_metadata_path(
    pair: IndexPair,
    index: DownloadIndex,
    *,
    level: str,  # "L1C" | "L2A"
    require_present: bool = True,
) -> Path:
    lvl = level.strip().upper()
    if lvl == "L1C":
        p = select_assets(
            pair,
            index,
            l1c_bands=(),
            need_l1c_tile_metadata=True,
            need_l2a_tile_metadata=False,
            need_scl_20m=False,
            require_present=require_present,
        ).l1c_tile_metadata
        if p is None:
            raise FileNotFoundError("Missing L1C tile_metadata")
        return p
    if lvl == "L2A":
        p = select_assets(
            pair,
            index,
            l1c_bands=(),
            need_l1c_tile_metadata=False,
            need_l2a_tile_metadata=True,
            need_scl_20m=False,
            require_present=require_present,
        ).l2a_tile_metadata
        if p is None:
            raise FileNotFoundError("Missing L2A tile_metadata")
        return p
    raise ValueError(f"level must be 'L1C' or 'L2A', got: {level!r}")


def select_scl_20m_path(
    pair: IndexPair,
    index: DownloadIndex,
    *,
    require_present: bool = True,
) -> Path:
    p = select_assets(
        pair,
        index,
        l1c_bands=(),
        need_l1c_tile_metadata=False,
        need_l2a_tile_metadata=False,
        need_scl_20m=True,
        require_present=require_present,
    ).scl_20m
    if p is None:
        raise FileNotFoundError("Missing SCL_20m")
    return p
