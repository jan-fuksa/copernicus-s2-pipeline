from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .http import CDSEHttpClient
from .nodes import NodeEntry, node_value_url


@dataclass(frozen=True)
class DownloadedFile:
    role: str
    path: Path
    band: str | None = None


def pair_dir(base: Path, *, tile_id: str, sensing_compact: str) -> Path:
    return base / f"tile={tile_id}" / f"sensing={sensing_compact}Z" / "files"


def download_node(
    client: CDSEHttpClient,
    *,
    product_id: str,
    node: NodeEntry,
    dst: Path,
    overwrite: bool,
    chunk_size: int,
) -> None:
    url = node_value_url(product_id, node.parts)
    client.stream_download(url, dst, overwrite=overwrite, chunk_size=chunk_size)
