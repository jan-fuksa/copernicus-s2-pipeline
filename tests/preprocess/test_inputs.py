import os
from pathlib import Path

import pytest

from s2pipe.preprocess.inputs import (
    load_download_index,
    iter_scenes,
    select_assets,
    DownloadIndex,
    IndexScene,
    SelectedAssets,
)


def _testdata_root() -> Path:
    env = os.environ.get("S2PIPE_TESTDATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parents[1] / "fixtures" / "single_tile").resolve()


@pytest.mark.integration
def test_load_iter_select_assets():
    root = _testdata_root()
    index_path = root / "meta" / "step1" / "index.json"
    if not index_path.exists():
        pytest.skip(f"Missing test data index.json at: {index_path}")

    assert index_path.is_file()
    d = load_download_index(index_path)
    assert isinstance(d, DownloadIndex)
    for item in iter_scenes(d):
        assert isinstance(item, IndexScene)
        required_roles = {
            "l1c.band.B01",
            "l1c.band.B02",
            "l1c.product_metadata",
            "l1c.tile_metadata",
            "l2a.scl_20m",
            "l2a.tile_metadata",
        }
        a = select_assets(
            item,
            d,
            required_roles=required_roles,
            require_present=True,
        )
        assert isinstance(a, SelectedAssets)
