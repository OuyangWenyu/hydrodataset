"""Tests for MultiDatasetReader."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from hydrodataset.multi_dataset_reader import MultiDatasetReader


# ── Construction ───────────────────────────────────────────────────────


def test_constructor_defaults():
    reader = MultiDatasetReader()
    assert isinstance(reader.datasets, list)
    assert len(reader.datasets) > 0
    assert reader.source == "local"
    assert reader.cache_dir.is_dir()


def test_constructor_custom_datasets():
    reader = MultiDatasetReader(datasets=["bull", "simbi"])
    assert reader.datasets == ["bull", "simbi"]


def test_constructor_invalid_dataset():
    with pytest.raises(ValueError, match="Unknown dataset"):
        MultiDatasetReader(datasets=["not_a_dataset"])


def test_constructor_s3_source():
    reader = MultiDatasetReader(datasets=["bull"], source="cloud")
    assert reader.source == "cloud"


# ── ID collection ─────────────────────────────────────────────────────


def test_collect_all_ids_caches(tmp_path):
    reader = MultiDatasetReader(
        cache_dir=str(tmp_path), datasets=["simbi"]
    )
    ids1 = reader.collect_all_ids()
    assert "simbi" in ids1
    assert len(ids1["simbi"]) > 0

    # Second call reads from cache
    ids2 = reader.collect_all_ids()
    assert ids1 == ids2

    cache_file = tmp_path / "multi_dataset_ids.json"
    assert cache_file.exists()


def test_collect_all_ids_force_refresh(tmp_path):
    reader = MultiDatasetReader(
        cache_dir=str(tmp_path), datasets=["simbi"]
    )
    reader.collect_all_ids()
    ids = reader.collect_all_ids(force_refresh=True)
    assert "simbi" in ids


# ── Deduplication ─────────────────────────────────────────────────────


def test_get_global_unique_ids_no_duplicates():
    reader = MultiDatasetReader(datasets=["bull", "simbi"])
    unique, dup = reader.get_global_unique_ids()
    assert len(unique) > 0
    # bull and simbi use different ID prefixes, so dup should be 0
    assert len(dup) == 0


def test_get_global_unique_ids_with_dict():
    id_map = {
        "ds_a": ["001", "002", "003"],
        "ds_b": ["003", "004"],
    }
    reader = MultiDatasetReader(datasets=[])  # empty, won't call collect
    unique, dup = reader.get_global_unique_ids(id_map)
    assert unique["001"] == "ds_a"
    assert unique["002"] == "ds_a"
    assert unique["003"] == "ds_a"  # first-come
    assert unique["004"] == "ds_b"
    assert "003" in dup
    assert set(dup["003"]) == {"ds_a", "ds_b"}


# ── Data reading ──────────────────────────────────────────────────────


@pytest.fixture
def mock_get_dataset():
    """Patch _get_dataset to return a mock that provides read_ts_xrdataset."""
    mock_ds = MagicMock()
    # Build a simple xarray Dataset with basin dim so .sel(basin=...).to_dataframe() works
    import xarray as xr

    ds = xr.Dataset(
        {
            "streamflow": xr.DataArray(
                [[1.0, 2.0], [3.0, 4.0]],
                dims=["time", "basin"],
                coords={
                    "time": pd.date_range("2000-01-01", periods=2),
                    "basin": ["001", "002"],
                },
            ),
        }
    )
    mock_ds.read_ts_xrdataset.return_value = ds
    return mock_ds


def test_read_data_keys_format(mock_get_dataset):
    """Output keys should be id@dataset."""
    reader = MultiDatasetReader(datasets=[])
    reader._get_dataset = lambda d: mock_get_dataset

    # Pre-populate id_mapping so the method finds the IDs
    reader.datasets = ["bull", "simbi"]
    with patch.object(reader, "collect_all_ids", return_value={
        "bull": ["001", "002"],
        "simbi": ["002"],
    }):
        data = reader.read_data(
            gage_ids=["001", "002"],
            t_range=["2000-01-01", "2000-01-02"],
        )

    # 001 only in bull, 002 in both
    assert "001@bull" in data
    assert "002@bull" in data
    assert "002@simbi" in data
    assert isinstance(data["001@bull"], pd.DataFrame)


def test_read_data_missing_id(mock_get_dataset):
    """Unknown IDs should be skipped with a warning, not crash."""
    reader = MultiDatasetReader(datasets=[])
    reader._get_dataset = lambda d: mock_get_dataset
    reader.datasets = ["bull"]

    with patch.object(reader, "collect_all_ids", return_value={
        "bull": ["001"],
    }):
        data = reader.read_data(
            gage_ids=["999"],
            t_range=["2000-01-01", "2000-01-02"],
        )

    assert len(data) == 0  # nothing found


# ── Integration test with real data ───────────────────────────────────

from tests._paths import needs_bull, needs_simbi, BULL_PATH, SIMBI_PATH
from tests.conftest import skip_if_ci


@pytest.mark.integration
@needs_bull
@needs_simbi
@skip_if_ci
def test_read_data_returns_real_timeseries():
    """End-to-end: collect IDs and read timeseries from bull + simbi."""
    reader = MultiDatasetReader(datasets=["bull", "simbi"])

    ids = reader.collect_all_ids()
    assert len(ids["bull"]) == 484
    assert len(ids["simbi"]) == 24

    data = reader.read_data(
        gage_ids=["BULL_10004"],
        t_range=["1997-07-01", "1997-07-10"],
    )
    assert "BULL_10004@bull" in data
    df = data["BULL_10004@bull"]
    assert not df.empty
    assert "streamflow" in df.columns
