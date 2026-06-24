"""Tests for LamaH datasets (lamah_ce, lamah_ice).

Follows the ADR 0001 unified data path resolution pattern:
- resolve_data_path() for path resolution
- Standard variable names + sources parameter for timeseries
- Ground-truth comparison against the raw data files

NOTE on data layout: both use registry path ``.`` (CAMELS-style); aqua_fetch
appends the reader class name, so raw files live under ``<root>/LamaHCE/...``
and ``<root>/LamaHIce/...``.
"""

import os

import numpy as np
import pandas as pd
import pytest

from hydrodataset import resolve_data_path, DatasetResolutionError
from hydrodataset.lamah_ce import LamahCe
from hydrodataset.lamah_ice import LamahIce

from tests._paths import (
    DATA_ROOT as data_path,
    LAMAH_CE_PATH,
    LAMAH_ICE_PATH,
    needs_lamah_ce,
    needs_lamah_ice,
)
from tests.conftest import skip_if_ci


# ── Resolver tests (no data needed) ───────────────────────────────────


def test_lamah_ce_in_reader_aliases():
    """lamah_ce must be a registered reader alias."""
    from hydrodataset.configs.data_resolver import READER_ALIASES

    assert "lamah_ce" in READER_ALIASES
    spec = READER_ALIASES["lamah_ce"]
    assert spec["class"] == "LamahCe"
    assert spec["module"] == "hydrodataset.lamah_ce"


def test_lamah_ice_in_reader_aliases():
    """lamah_ice must be a registered reader alias."""
    from hydrodataset.configs.data_resolver import READER_ALIASES

    assert "lamah_ice" in READER_ALIASES
    spec = READER_ALIASES["lamah_ice"]
    assert spec["class"] == "LamahIce"
    assert spec["module"] == "hydrodataset.lamah_ice"


def test_resolve_lamah_ce_returns_absolute_path():
    """resolve_data_path('lamah_ce') returns an absolute path."""
    try:
        p = resolve_data_path("lamah_ce")
        assert os.path.isabs(p), f"Expected absolute path, got {p}"
    except DatasetResolutionError:
        pytest.skip("lamah_ce data not available")


def test_resolve_lamah_ice_returns_absolute_path():
    """resolve_data_path('lamah_ice') returns an absolute path."""
    try:
        p = resolve_data_path("lamah_ice")
        assert os.path.isabs(p), f"Expected absolute path, got {p}"
    except DatasetResolutionError:
        pytest.skip("lamah_ice data not available")


# ── LamaH-CE static attribute test ────────────────────────────────────


@needs_lamah_ce
@skip_if_ci
def test_read_lamah_ce_attr_xrdataset():
    """read_attr_xrdataset values match the raw Catchment_attributes.csv."""
    ds = LamahCe(LAMAH_CE_PATH)
    p_mean_1 = ds.read_attr_xrdataset(gage_id_lst=["1"], var_lst=["p_mean"])[
        "p_mean"
    ].values
    area_1 = ds.read_attr_xrdataset(gage_id_lst=["1"], var_lst=["area"])["area"].values
    # Ground-truth from raw CSV (semicolon-separated; ``area`` maps to area_calc)
    csv_path = os.path.join(
        data_path,
        "LamaHCE",
        "A_basins_total_upstrm",
        "1_attributes",
        "Catchment_attributes.csv",
    )
    df = pd.read_csv(csv_path, sep=";")
    row = df[df["ID"] == 1]
    assert np.isclose(p_mean_1, row["p_mean"].values[0], rtol=1e-6), (
        f"p_mean mismatch: {p_mean_1} vs {row['p_mean'].values[0]}"
    )
    assert np.isclose(area_1, row["area_calc"].values[0], rtol=1e-6), (
        f"area mismatch: {area_1} vs {row['area_calc'].values[0]}"
    )


# ── LamaH-CE dynamic (timeseries) test ────────────────────────────────


# Uses standard variable name + sources parameter (new ADR 0001 pattern).
@needs_lamah_ce
@skip_if_ci
def test_read_lamah_ce_timeseries_xrdataset():
    """read_ts_xrdataset streamflow matches the raw gauge timeseries file."""
    ds = LamahCe(LAMAH_CE_PATH)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["1"],
        var_lst=["streamflow"],  # standard name
        t_range=["2000-06-01", "2000-06-01"],
        sources={"streamflow": "observations"},  # observed discharge (default)
    )
    result_1 = ts_data["streamflow"].values.flatten()[0]
    # Ground-truth from raw gauge timeseries CSV (semicolon-separated; qobs in m^3/s)
    file_path = os.path.join(
        data_path,
        "LamaHCE",
        "D_gauges",
        "2_timeseries",
        "daily",
        "ID_1.csv",
    )
    df = pd.read_csv(file_path, sep=";")
    result_2 = df[(df["YYYY"] == 2000) & (df["MM"] == 6) & (df["DD"] == 1)][
        "qobs"
    ].values[0]
    assert np.isclose(
        result_1, result_2, rtol=1e-6
    ), f"Expected {result_2}, got {result_1}"


# ── LamaH-Ice static attribute test ───────────────────────────────────


@needs_lamah_ice
@skip_if_ci
def test_read_lamah_ice_attr_xrdataset():
    """read_attr_xrdataset values match the raw Catchment_attributes.csv."""
    ds = LamahIce(LAMAH_ICE_PATH)
    p_mean_1 = ds.read_attr_xrdataset(gage_id_lst=["1"], var_lst=["p_mean"])[
        "p_mean"
    ].values
    area_1 = ds.read_attr_xrdataset(gage_id_lst=["1"], var_lst=["area"])["area"].values
    # Ground-truth from raw CSV (semicolon-separated; lowercase ``id`` column,
    # ``area`` maps to area_calc). Note the double-nested lamah_ice/lamah_ice path.
    csv_path = os.path.join(
        data_path,
        "LamaHIce",
        "lamah_ice",
        "lamah_ice",
        "A_basins_total_upstrm",
        "1_attributes",
        "Catchment_attributes.csv",
    )
    df = pd.read_csv(csv_path, sep=";")
    row = df[df["id"] == 1]
    assert np.isclose(p_mean_1, row["p_mean"].values[0], rtol=1e-6), (
        f"p_mean mismatch: {p_mean_1} vs {row['p_mean'].values[0]}"
    )
    assert np.isclose(area_1, row["area_calc"].values[0], rtol=1e-6), (
        f"area mismatch: {area_1} vs {row['area_calc'].values[0]}"
    )


# ── LamaH-Ice dynamic (timeseries) test ───────────────────────────────


# Uses standard variable name + sources parameter (new ADR 0001 pattern).
@needs_lamah_ice
@skip_if_ci
def test_read_lamah_ice_timeseries_xrdataset():
    """read_ts_xrdataset streamflow matches the raw gauge timeseries file."""
    ds = LamahIce(LAMAH_ICE_PATH)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["1"],
        var_lst=["streamflow"],  # standard name
        t_range=["2013-02-11", "2013-02-11"],  # station 1 daily record starts 2004
        sources={"streamflow": "lamah_ice"},  # observed discharge (default)
    )
    result_1 = ts_data["streamflow"].values.flatten()[0]
    # Ground-truth from raw gauge timeseries CSV (semicolon-separated; qobs in m^3/s)
    file_path = os.path.join(
        data_path,
        "LamaHIce",
        "lamah_ice",
        "lamah_ice",
        "D_gauges",
        "2_timeseries",
        "daily",
        "ID_1.csv",
    )
    df = pd.read_csv(file_path, sep=";")
    result_2 = df[(df["YYYY"] == 2013) & (df["MM"] == 2) & (df["DD"] == 11)][
        "qobs"
    ].values[0]
    assert np.isclose(
        result_1, result_2, rtol=1e-6
    ), f"Expected {result_2}, got {result_1}"
