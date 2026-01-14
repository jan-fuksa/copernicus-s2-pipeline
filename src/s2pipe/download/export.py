from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Sequence

import pandas as pd


def _safe_col_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def manifest_to_dataframe(
    manifest_rows: Sequence[Any],
    *,
    flatten_local_paths: bool = True,
    local_paths_prefix: str = "path__",
    keep_local_paths_json: bool = False,
) -> pd.DataFrame:
    """
    Convert a list of dict rows OR dataclasses to a 2D table.

    This function targets the *legacy* "rows" shape (ManifestRow-like):
    { tile_id, sensing_start, ..., local_paths: {k: path, ...} }.

    For the new structured manifest (DownloadManifest), prefer exporting from
    SceneEntry structures (see `scenes_to_dataframe` below).
    """
    rows: list[dict[str, Any]] = []
    for r in manifest_rows:
        if is_dataclass(r):
            d = asdict(r)
        elif isinstance(r, dict):
            d = dict(r)
        else:
            raise TypeError(f"Unsupported row type: {type(r)}")
        rows.append(d)

    all_lp_keys: list[str] = []
    if flatten_local_paths:
        key_set: set[str] = set()
        for d in rows:
            lp = d.get("local_paths") or {}
            if isinstance(lp, Mapping):
                key_set.update(lp.keys())
        all_lp_keys = sorted(key_set)

    table_rows: list[dict[str, Any]] = []
    for d in rows:
        lp = d.get("local_paths") or {}
        base = {k: v for k, v in d.items() if k != "local_paths"}

        if keep_local_paths_json:
            base["local_paths_json"] = (
                json.dumps(lp, ensure_ascii=False) if isinstance(lp, Mapping) else None
            )

        if flatten_local_paths:
            if isinstance(lp, Mapping):
                for k in all_lp_keys:
                    col = local_paths_prefix + _safe_col_name(str(k))
                    base[col] = lp.get(k)
            else:
                for k in all_lp_keys:
                    col = local_paths_prefix + _safe_col_name(str(k))
                    base[col] = None

        table_rows.append(base)

    df = pd.DataFrame(table_rows)
    return df


def scenes_to_dataframe(scenes: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert manifest['scenes'] (structured SceneEntry dicts) to a 2D table.
    - One row per scene
    - Columns include key fields + l1c/l2a metadata + file paths (flattened by role)
    """
    rows: list[dict[str, Any]] = []
    for p in scenes:
        row: dict[str, Any] = {}
        key = p.get("key", {})
        row["tile_id"] = key.get("tile_id")
        row["sensing_start_utc"] = key.get("sensing_start_utc")

        l1c = p.get("l1c", {})
        l2a = p.get("l2a", {})
        for prefix, obj in (("l1c", l1c), ("l2a", l2a)):
            row[f"{prefix}_product_id"] = obj.get("product_id")
            row[f"{prefix}_product_name"] = obj.get("product_name")
            row[f"{prefix}_baseline"] = obj.get("baseline")
            row[f"{prefix}_rel_orbit"] = obj.get("rel_orbit")
            if prefix == "l1c":
                row["cloud_cover"] = obj.get("cloud_cover")
                row["coverage_ratio"] = obj.get("coverage_ratio")

        # Files: flatten by role/band
        def add_files(prefix: str, files_obj: dict[str, Any]) -> None:
            items = files_obj.get("items", []) or []
            for it in items:
                role = (it.get("role") or "file").strip()
                band = it.get("band")
                k = f"{prefix}_{role}"
                if band:
                    k = f"{k}_{band}"
                row[k] = it.get("path")

        add_files("l1c", p.get("files_l1c", {}))
        add_files("l2a", p.get("files_l2a", {}))

        rows.append(row)

    return pd.DataFrame(rows)


def export_table(
    df: pd.DataFrame, *, csv_path: str | None = None, xlsx_path: str | None = None
) -> None:
    if csv_path:
        df.to_csv(csv_path, index=False)
    if xlsx_path:
        df.to_excel(xlsx_path, index=False)
