from __future__ import annotations

from typing import Sequence

from .nodes import NodeEntry
from s2pipe.download.cfg import SelectionConfig


def _endswith_any(name: str, suffixes: Sequence[str]) -> bool:
    name_l = name.lower()
    return any(name_l.endswith(s.lower()) for s in suffixes)


def select_assets_l1c(
    index: Sequence[NodeEntry], sel: SelectionConfig
) -> list[NodeEntry]:
    out: list[NodeEntry] = []

    want_bands = set(sel.l1c_bands)
    for n in index:
        if n.is_dir:
            continue
        fn = n.parts[-1]
        if sel.l1c_tile_metadata and fn == "MTD_TL.xml":
            out.append(n)
            continue
        if sel.l1c_tile_metadata and fn == "MTD_MSIL1C.xml":
            out.append(n)
            continue

        # JP2 bands: typical filename contains "_B02.jp2" etc.
        for b in want_bands:
            if fn.endswith(f"_{b}.jp2"):
                out.append(n)
                break

    return out


def select_assets_l2a(
    index: Sequence[NodeEntry], sel: SelectionConfig
) -> list[NodeEntry]:
    out: list[NodeEntry] = []

    for n in index:
        if n.is_dir:
            continue
        fn = n.parts[-1]

        if sel.l2a_tile_metadata and fn == "MTD_TL.xml":
            out.append(n)
            continue

        if sel.l2a_scl_20m and fn.endswith("_SCL_20m.jp2"):
            out.append(n)
            continue

        if sel.l2a_aot_20m and fn.endswith("_AOT_20m.jp2"):
            out.append(n)
            continue

        if sel.l2a_wvp_20m and fn.endswith("_WVP_20m.jp2"):
            out.append(n)
            continue

    return out
