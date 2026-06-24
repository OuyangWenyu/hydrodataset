"""Tests for other public datasets (bull, hysets, estreams, simbi).

Each dataset gets two value-comparison tests:
- read_attr_xrdataset() against raw attribute CSVs
- read_ts_xrdataset() against raw timeseries files

Follows the same pattern as ``test_camels_series.py`` (camels_aus/camels_ch).
"""

import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydrodataset.bull import BULL
from hydrodataset.simbi import simbi
from hydrodataset.estreams import Estreams
from hydrodataset.hysets import Hysets

from tests._paths import (
    DATA_ROOT as data_path,
    BULL_PATH,
    SIMBI_PATH,
    ESTREAMS_PATH,
    HYSETS_PATH,
    needs_bull,
    needs_simbi,
    needs_estreams,
    needs_hysets,
)
from tests.conftest import skip_if_ci


# ── BULL ───────────────────────────────────────────────────────────────

_BULL_GAUGE = "BULL_10004"
_BULL_KNOWN_STATION = "10004"  # bare id used in expected values


@needs_bull
@skip_if_ci
def test_read_bull_attr_xrdataset():
    """read_attr_xrdataset values match the raw BULL attribute CSVs."""
    ds = BULL(BULL_PATH)
    p_mean_1 = ds.read_attr_xrdataset(
        gage_id_lst=[_BULL_GAUGE], var_lst=["p_mean"]
    )["p_mean"].values
    area_1 = ds.read_attr_xrdataset(
        gage_id_lst=[_BULL_GAUGE], var_lst=["area"]
    )["area"].values
    # Ground-truth: p_mean in attributes_caravan_.csv, area in attributes_other_ss.csv
    attr_dir = os.path.join(data_path, "Bull", "attributes")
    p_df = pd.read_csv(
        os.path.join(attr_dir, "attributes_caravan_.csv"), dtype={"gauge_id": str}
    )
    a_df = pd.read_csv(
        os.path.join(attr_dir, "attributes_other_ss.csv"), dtype={"gauge_id": str}
    )
    expected_p = p_df[p_df["gauge_id"] == _BULL_GAUGE]["p_mean"].values[0]
    expected_a = a_df[a_df["gauge_id"] == _BULL_GAUGE]["area"].values[0]
    assert np.isclose(p_mean_1, expected_p, rtol=1e-6), (
        f"p_mean mismatch: {p_mean_1} vs {expected_p}"
    )
    assert np.isclose(area_1, expected_a, rtol=1e-6), (
        f"area mismatch: {area_1} vs {expected_a}"
    )


@needs_bull
@skip_if_ci
def test_read_bull_timeseries_xrdataset():
    """read_ts_xrdataset streamflow matches the raw BULL timeseries CSV."""
    ds = BULL(BULL_PATH)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=[_BULL_GAUGE],
        var_lst=["streamflow"],
        t_range=["1997-07-06", "1997-07-06"],
    )
    result_1 = ts_data["streamflow"].values.flatten()[0]
    # Ground-truth: streamflow_<bare_id>.csv in the streamflow subdirectory
    file_path = os.path.join(
        data_path,
        "Bull",
        "timeseries",
        "timeseries",
        "csv",
        "streamflow",
        f"streamflow_{_BULL_KNOWN_STATION}.csv",
    )
    df = pd.read_csv(file_path)
    result_2 = df[df["date"].astype(str).str[:10] == "1997-07-06"][
        "streamflow"
    ].values[0]
    assert np.isclose(
        result_1, result_2, rtol=1e-6
    ), f"Expected {result_2}, got {result_1}"


# ── SIMBI ──────────────────────────────────────────────────────────────

_SIMBI_GAUGE = "001"  # aqua_fetch returns bare id
_SIMBI_RAW_GAUGE = "Q-001"  # raw CSVs use Q- prefixed gauge code


@needs_simbi
@skip_if_ci
def test_read_simbi_attr_xrdataset():
    """read_attr_xrdataset values match the raw SIMBI attribute CSVs."""
    ds = simbi(SIMBI_PATH)
    p_mean_1 = ds.read_attr_xrdataset(
        gage_id_lst=[_SIMBI_GAUGE], var_lst=["p_mean"]
    )["p_mean"].values
    area_1 = ds.read_attr_xrdataset(
        gage_id_lst=[_SIMBI_GAUGE], var_lst=["area"]
    )["area"].values
    # Ground-truth: p_mean (P column) in average.csv, area in location_and_topography.csv
    attr_base = os.path.join(data_path, "Simbi", "03_SIMBI_ATTRIBUTE")
    p_df = pd.read_csv(
        os.path.join(
            attr_base, "01_CLIMATIC_SIGNATURE", "01_MONTHLY", "average.csv"
        )
    )
    a_df = pd.read_csv(
        os.path.join(attr_base, "02_OTHERS", "location_and_topography.csv")
    )
    expected_p = p_df[p_df["Gauge_code"] == _SIMBI_RAW_GAUGE]["P"].values[0]
    expected_a = a_df[a_df["Gauge_code"] == _SIMBI_RAW_GAUGE]["Area"].values[0]
    assert np.isclose(p_mean_1, expected_p, rtol=1e-6), (
        f"p_mean mismatch: {p_mean_1} vs {expected_p}"
    )
    assert np.isclose(area_1, expected_a, rtol=1e-6), (
        f"area mismatch: {area_1} vs {expected_a}"
    )


@needs_simbi
@skip_if_ci
def test_read_simbi_timeseries_xrdataset():
    """read_ts_xrdataset streamflow matches the raw SIMBI timeseries CSV."""
    ds = simbi(SIMBI_PATH)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=[_SIMBI_GAUGE],
        var_lst=["streamflow"],
        t_range=["1930-04-01", "1930-04-01"],
    )
    result_1 = ts_data["streamflow"].values.flatten()[0]
    # Ground-truth from raw daily streamflow CSV
    file_path = os.path.join(
        data_path,
        "Simbi",
        "00_SIMBI_OBSERVED_DATA",
        "02_DAILY_STREAMFLOW",
        f"Q_{_SIMBI_GAUGE}.csv",
    )
    df = pd.read_csv(file_path)
    result_2 = df[df["Date"] == "1930-04-01"]["Q"].values[0]
    assert np.isclose(
        result_1, result_2, rtol=1e-6
    ), f"Expected {result_2}, got {result_1}"


# ── EStreams ───────────────────────────────────────────────────────────

_ESTREAMS_GAUGE = "AT000001"


@needs_estreams
@skip_if_ci
def test_read_estreams_attr_xrdataset():
    """read_attr_xrdataset values match the raw EStreams attribute files."""
    ds = Estreams(ESTREAMS_PATH)
    p_mean_1 = ds.read_attr_xrdataset(
        gage_id_lst=[_ESTREAMS_GAUGE], var_lst=["p_mean"]
    )["p_mean"].values
    area_1 = ds.read_attr_xrdataset(
        gage_id_lst=[_ESTREAMS_GAUGE], var_lst=["area"]
    )["area"].values
    # Ground-truth: p_mean in hydroclimatic_signatures csv, area in gauging_stations csv
    es_dir = os.path.join(data_path, "EStreams", "EStreams", "EStreams")
    sig_df = pd.read_csv(
        os.path.join(es_dir, "hydroclimatic_signatures",
                     "estreams_hydrometeo_signatures.csv"),
        dtype={"basin_id": str},
    )
    gauge_df = pd.read_csv(
        os.path.join(es_dir, "streamflow_gauges",
                     "estreams_gauging_stations.csv"),
        dtype={"basin_id": str},
    )
    expected_p = sig_df[sig_df["basin_id"] == _ESTREAMS_GAUGE]["p_mean"].values[0]
    expected_a = gauge_df[gauge_df["basin_id"] == _ESTREAMS_GAUGE]["area_estreams"].values[0]
    assert np.isclose(p_mean_1, expected_p, rtol=1e-6), (
        f"p_mean mismatch: {p_mean_1} vs {expected_p}"
    )
    assert np.isclose(area_1, expected_a, rtol=1e-6), (
        f"area mismatch: {area_1} vs {expected_a}"
    )


@needs_estreams
@skip_if_ci
def test_read_estreams_timeseries_xrdataset():
    """read_ts_xrdataset temperature_mean matches the raw meteorology CSV."""
    ds = Estreams(ESTREAMS_PATH)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=[_ESTREAMS_GAUGE],
        var_lst=["temperature_mean"],
        t_range=["1986-10-01", "1986-10-01"],
    )
    result_1 = ts_data["temperature_mean"].values.flatten()[0]
    # Ground-truth from raw per-station meteorology CSV
    file_path = os.path.join(
        data_path,
        "EStreams",
        "EStreams",
        "EStreams",
        "meteorology",
        f"estreams_meteorology_{_ESTREAMS_GAUGE}.csv",
    )
    df = pd.read_csv(file_path)
    result_2 = df[df["date"] == "1986-10-01"]["t_mean"].values[0]
    assert np.isclose(
        result_1, result_2, rtol=1e-6
    ), f"Expected {result_2}, got {result_1}"


# ── HYSETS ─────────────────────────────────────────────────────────────

# aqua_fetch uses 1-based string IDs ("1", "2", ...) that map to
# 0-based watershed indices in the raw NetCDF files.
_HYSETS_GAUGE = "1"
_HYSETS_RAW_WS = 0


@needs_hysets
@skip_if_ci
def test_read_hysets_attr_xrdataset():
    """read_attr_xrdataset area matches the raw HYSETS NetCDF drainage_area."""
    ds = Hysets(HYSETS_PATH)
    area_1 = ds.read_attr_xrdataset(
        gage_id_lst=[_HYSETS_GAUGE], var_lst=["area"]
    )["area"].values
    # Ground-truth from raw multi-station NetCDF (QC stations)
    nc_path = os.path.join(
        data_path, "HYSETS", "HYSETS_2023_update_QC_stations.nc"
    )
    raw_ds = xr.open_dataset(nc_path)
    expected_a = raw_ds["drainage_area"].values[_HYSETS_RAW_WS]
    raw_ds.close()
    assert np.isclose(area_1, expected_a, rtol=1e-6), (
        f"area mismatch: {area_1} vs {expected_a}"
    )


@needs_hysets
@skip_if_ci
def test_read_hysets_timeseries_xrdataset():
    """read_ts_xrdataset streamflow matches the raw HYSETS NetCDF discharge."""
    ds = Hysets(HYSETS_PATH)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=[_HYSETS_GAUGE],
        var_lst=["streamflow"],
        t_range=["1986-07-02", "1986-07-02"],
        sources={"streamflow": "observations_cms"},
    )
    result_1 = ts_data["streamflow"].values.flatten()[0]
    # Ground-truth from raw NetCDF (watershed=0, time="1986-07-02")
    nc_path = os.path.join(
        data_path, "HYSETS", "HYSETS_2023_update_QC_stations.nc"
    )
    raw_ds = xr.open_dataset(nc_path)
    result_2 = float(
        raw_ds["discharge"].sel(watershed=_HYSETS_RAW_WS, time="1986-07-02").values
    )
    raw_ds.close()
    assert np.isclose(
        result_1, result_2, rtol=1e-6
    ), f"Expected {result_2}, got {result_1}"
