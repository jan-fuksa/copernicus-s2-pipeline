from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence
from urllib.parse import quote

from .http import CDSEHttpClient


DOWNLOAD_ODATA_BASE = "https://download.dataspace.copernicus.eu/odata/v1"


@dataclass(frozen=True)
class NodeEntry:
    parts: tuple[str, ...]
    is_dir: bool


def nodes_url(product_id: str, parts: Optional[Sequence[str]] = None) -> str:
    if not parts:
        return f"{DOWNLOAD_ODATA_BASE}/Products({product_id})/Nodes"
    segs = "".join([f"/Nodes({quote(p, safe='_.-~')})" for p in parts])
    return f"{DOWNLOAD_ODATA_BASE}/Products({product_id}){segs}/Nodes"


def node_value_url(product_id: str, parts: Sequence[str]) -> str:
    segs = "".join([f"/Nodes({quote(p, safe='_.-~')})" for p in parts])
    return f"{DOWNLOAD_ODATA_BASE}/Products({product_id}){segs}/$value"


_NODES_INDEX_CACHE: dict[str, list[NodeEntry]] = {}


def list_nodes(client: CDSEHttpClient, product_id: str, parts: Optional[Sequence[str]] = None) -> list[dict[str, Any]]:
    url = nodes_url(product_id, parts)
    js = client.get_json(url)

    items = js.get("result")
    if items is None:
        items = js.get("value")
    if items is None:
        raise RuntimeError(f"Unexpected Nodes listing schema at {url}: keys={list(js.keys())}")
    if not isinstance(items, list):
        raise RuntimeError(f"Unexpected Nodes listing payload at {url}: type(items)={type(items)}")
    return items


def index_product_nodes(
    client: CDSEHttpClient,
    product_id: str,
    *,
    skip_dir_names: set[str],
    skip_prefixes: tuple[tuple[str, ...], ...],
    max_dirs_to_visit: int,
    enable_cache: bool,
) -> list[NodeEntry]:
    if enable_cache and product_id in _NODES_INDEX_CACHE:
        return _NODES_INDEX_CACHE[product_id]

    def should_skip_descend(parts: tuple[str, ...]) -> bool:
        if parts and parts[-1] in skip_dir_names:
            return True
        for pref in skip_prefixes:
            if pref and parts[: len(pref)] == pref:
                return True
        return False

    out: list[NodeEntry] = []
    stack: list[tuple[str, ...]] = [tuple()]
    visited: set[tuple[str, ...]] = set()
    dirs_visited = 0

    while stack:
        parent_parts = stack.pop()
        if parent_parts in visited:
            continue
        visited.add(parent_parts)

        children = list_nodes(client, product_id, parent_parts if parent_parts else None)

        for item in children:
            name = item.get("Name") or item.get("Id")
            if not isinstance(name, str) or not name:
                continue

            child_parts = parent_parts + (name,)
            content_len = item.get("ContentLength")
            children_num = item.get("ChildrenNumber")

            if isinstance(content_len, int):
                is_dir = (content_len == 0)
            else:
                is_dir = isinstance(children_num, int) and children_num > 0

            out.append(NodeEntry(parts=child_parts, is_dir=is_dir))

            if is_dir:
                if should_skip_descend(child_parts):
                    continue
                stack.append(child_parts)
                dirs_visited += 1
                if dirs_visited > max_dirs_to_visit:
                    raise RuntimeError(
                        f"Aborting indexing: too many directories visited ({dirs_visited}). product_id={product_id}"
                    )

    if enable_cache:
        _NODES_INDEX_CACHE[product_id] = out
    return out
