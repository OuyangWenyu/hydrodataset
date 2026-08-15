import xarray as xr
import numpy as np
import pandas as pd
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from hydrodataset import resolve_data_path, StandardVariable
from hydrodataset.camelsh import Camelsh
from hydrodataset.camels_aus import CamelsAus
from hydrodataset.camels_cl import CamelsCl
from hydrodataset.camels_dk import CamelsDk
from hydrodataset.camels_col import CamelsCol
from hydrodataset.camels_se import CamelsSe
from hydrodataset.camelsh_kr import CamelshKr
from hydrodataset.camels_gb import CamelsGb
from hydrodataset.camels_fi import CamelsFi
from hydrodataset.camels_pe import CamelsPe
from hydrodataset.camels_lux import CamelsLux
from hydrodataset.camels_nz import CamelsNz
from hydrodataset.camels_de import CamelsDe
from hydrodataset.camels_fr import CamelsFr
from hydrodataset.camels_ch import CamelsCh
from hydrodataset.camels_us import CamelsUs
from hydrodataset.camels_ind import CamelsInd
from hydrodataset.camels_br import CamelsBr

from tests._paths import (
    DATA_ROOT as data_path,
    CAMELSH_PATH,
    CAMELS_AUS_PATH,
    CAMELS_CL_PATH,
    CAMELS_DK_PATH,
    CAMELS_COL_PATH,
    CAMELS_SE_PATH,
    CAMELSH_KR_PATH,
    CAMELS_GB_PATH,
    CAMELS_FI_PATH,
    CAMELS_PE_PATH,
    CAMELS_LUX_PATH,
    CAMELS_NZ_PATH,
    CAMELS_DE_PATH,
    CAMELS_FR_PATH,
    CAMELS_CH_PATH,
    CAMELS_US_PATH,
    CAMELS_IND_PATH,
    CAMELS_BR_PATH,
    needs_camelsh,
    needs_camels_aus,
    needs_camels_cl,
    needs_camels_dk,
    needs_camels_col,
    needs_camels_se,
    needs_camelsh_kr,
    needs_camels_gb,
    needs_camels_fi,
    needs_camels_pe,
    needs_camels_lux,
    needs_camels_nz,
    needs_camels_de,
    needs_camels_fr,
    needs_camels_ch,
    needs_camels_us,
    needs_camels_ind,
    needs_camels_br,
)
from tests.conftest import skip_if_ci

# "Test whether read_attr_xrdataset() from camelsh correctly reads .nc files and returns a list of watershed ID strings."
@needs_camelsh
@skip_if_ci
def test_read_attr_xrdataset():
    ds = Camelsh(CAMELSH_PATH)
    result_1 = ds.read_attr_xrdataset(gage_id_lst=["01011000"], var_lst=["p_mean"])[
        "p_mean"
    ].values
    csv_path = os.path.join(
        data_path,
        "CAMELSH",
        "attributes",
        "attributes",
        "attributes_nldas2_climate.csv",
    )
    df = pd.read_csv(csv_path)
    result_2 = df[df["STAID"] == 1011000]["p_mean"].values[0]
    assert result_1 == result_2


# "Test whether read_ts_xrdataset() from camelsh correctly reads .nc files and returns a list of watershed ID strings."
# Uses standard variable name + sources parameter (new ADR 0001 pattern).
@needs_camelsh
@skip_if_ci
def test_read_timeseries_xrdataset():
    ds = Camelsh(CAMELSH_PATH)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["01011000"],
        var_lst=["potential_evapotranspiration"],  # standard name
        t_range=["1980-01-01", "1980-01-01"],
        sources={"potential_evapotranspiration": "nldas"},  # select NLDAS source
    )
    station_data = ts_data["potential_evapotranspiration"]
    result_1 = station_data.values.flatten()
    file_path = os.path.join(
        data_path,
        "CAMELSH",
        "timeseries",
        "Data",
        "CAMELSH",
        "timeseries",
        "01011000.nc",
    )
    ds = xr.open_dataset(file_path)
    pet_data = ds["PotEvap"]
    result_2 = pet_data.isel(DateTime=slice(0, 24)).values
    assert np.allclose(
        result_1, result_2, rtol=1e-6
    ), f"PET values mismatch between cache and raw NC file"


# "Test whether read_attr_xrdataset() from camels_aus correctly reads .nc files and returns a list of watershed ID strings."
@needs_camels_aus
@skip_if_ci
def test_read_camels_aus_attr_xrdataset():
    # Use resolver chain: settings + registry -> absolute URI
    ds = CamelsAus(CAMELS_AUS_PATH)
    # Use standard variable name registered in _subclass_static_definitions
    result_1 = ds.read_attr_xrdataset(gage_id_lst=["912105A"], var_lst=["anngro_mega"])[
        "anngro_mega"
    ].values
    # Ground-truth comparison from raw CSV
    csv_path = os.path.join(
        data_path,
        "CAMELS_AUS",
        "04_attributes",
        "04_attributes",
        "CatchmentAttributes_05_Other.csv",
    )
    df = pd.read_csv(csv_path)
    result_2 = df[df["station_id"] == "912105A"]["anngro_mega"].values[0]
    assert result_1 == result_2


# "Test whether read_ts_xrdataset() from camels_aus correctly reads .nc files and returns a list of watershed ID strings."
# Uses standard variable name + sources parameter (new ADR 0001 pattern).
@needs_camels_aus
@skip_if_ci
def test_read_camels_aus_timeseries_xrdataset():
    ds = CamelsAus(CAMELS_AUS_PATH)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["912105A"],
        var_lst=["temperature_min"],  # standard name
        t_range=["1980-01-04", "1980-01-04"],
        sources={"temperature_min": ["silo"]},  # select SILO source
    )
    # Access data via standard name (not raw specific_name)
    station_data = ts_data["temperature_min"]
    result_1 = station_data.values.flatten()[0]
    # Ground-truth comparison from raw CSV
    file_path = os.path.join(
        data_path,
        "CAMELS_AUS",
        "05_hydrometeorology",
        "05_hydrometeorology",
        "03_Other",
        "SILO",
        "tmin_SILO.csv",
    )
    csv_df = pd.read_csv(file_path)
    result_2 = csv_df.loc[
        (csv_df["year"] == 1980) & (csv_df["month"] == 1) & (csv_df["day"] == 4),
        "912105A",
    ].values[0]
    assert np.isclose(
        result_1, result_2, rtol=1e-6
    ), f"Expected {result_2}, got {result_1}"



# "Test whether read_attr_xrdataset() from camels_cl correctly reads .nc files and returns a list of watershed ID strings."
@needs_camels_cl
@skip_if_ci
def test_read_camels_cl_attr_xrdataset():
    ds = CamelsCl(CAMELS_CL_PATH)
    # NOTE: CAMELS_CL cache stores all attributes as strings from the raw text file,
    # so we pass to_numeric=False to avoid factorizing string values to integer codes.
    result_1 = ds.read_attr_xrdataset(
        gage_id_lst=["1021001"], var_lst=["elev_mean"], to_numeric=False
    )["elev_mean"].values
    file_path = os.path.join(
        data_path,
        "CAMELS_CL",
        "1_CAMELScl_attributes",
        "1_CAMELScl_attributes.txt",
    )
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if "elev_mean" in line:
                parts = line.split("elev_mean", 1)[1].split()
                result_2 = parts[6].strip('"')
                break
    assert result_1 == result_2


# "Test whether read_ts_xrdataset() from camels_cl correctly reads .nc files and returns a list of watershed ID strings."
# Uses standard variable name + sources parameter (new ADR 0001 pattern).
@needs_camels_cl
@skip_if_ci
def test_read_camels_cl_timeseries_xrdataset():
    ds = CamelsCl(CAMELS_CL_PATH)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["1021001"],
        var_lst=["streamflow"],  # standard name
        t_range=["1981-01-04", "1981-01-04"],
        sources={"streamflow": "observations"},  # the default source
    )
    station_data = ts_data["streamflow"]
    result_1 = station_data.values.flatten()[0]
    file_path = os.path.join(
        data_path,
        "CAMELS_CL",
        "2_CAMELScl_streamflow_m3s",
        "2_CAMELScl_streamflow_m3s.txt",
    )
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if "1981-01-04" in line:
                parts = line.split("1981-01-04", 1)[1].split()
                result_2 = parts[8].strip('"')
                break
    assert np.isclose(
        float(result_1), float(result_2), rtol=1e-6
    ), f"Expected {result_2}, got {result_1}"



# "Test whether read_attr_xrdataset() from camels_dk correctly reads .nc files and returns a list of watershed ID strings."
@needs_camels_dk
@skip_if_ci
def test_read_camels_dk_attr_xrdataset():
    ds = CamelsDk(CAMELS_DK_PATH)
    result_1 = ds.read_attr_xrdataset(gage_id_lst=["12431077"], var_lst=["p_mean"])[
        "p_mean"
    ].values
    csv_path = os.path.join(
        data_path,
        "CAMELS_DK",
        "CAMELS_DK_climate.csv",
    )
    df = pd.read_csv(csv_path)
    result_2 = df[df["catch_id"] == 12431077]["p_mean"].values[0]
    assert result_1 == result_2


# "Test whether read_ts_xrdataset() from camels_dk correctly reads .nc files and returns a list of watershed ID strings."
# Uses standard variable name + sources parameter (new ADR 0001 pattern).
@needs_camels_dk
@skip_if_ci
def test_read_camels_dk_timeseries_xrdataset():
    ds = CamelsDk(CAMELS_DK_PATH)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["12431077"],
        var_lst=["precipitation"],  # standard name
        t_range=["1990-01-05", "1990-01-05"],
        sources={"precipitation": "dmi"},  # select DMI source
    )
    station_data = ts_data["precipitation"]
    result_1 = station_data.values.flatten()
    file_path = os.path.join(
        data_path,
        "CAMELS_DK",
        "Gauged_catchments",
        "Gauged_catchments",
        "CAMELS_DK_obs_based_12431077.csv",
    )
    ds = pd.read_csv(file_path)
    result_2 = ds[ds["time"] == "1990-01-05"]["precipitation"].values[0]
    assert np.isclose(
        result_1, result_2, rtol=1e-6
    ), f"Expected {result_2}, got {result_1}"



# "Test whether read_attr_xrdataset() from camels_col correctly reads .nc files and returns a list of watershed ID strings."
@needs_camels_col
@skip_if_ci
def test_read_camels_col_attr_xrdataset():
    ds = CamelsCol(CAMELS_COL_PATH)
    result_1 = ds.read_attr_xrdataset(gage_id_lst=["11027030"], var_lst=["q_mean"])[
        "q_mean"
    ].values
    csv_path = os.path.join(
        data_path,
        "CAMELS_COL",
        "09_CAMELS_COL_Hydrological_signatures.xlsx",
    )
    df = pd.read_excel(csv_path)
    result_2 = df[df["gauge_id"] == 11027030]["q_mean"].values[0]
    assert result_1 == result_2


# "Test whether read_ts_xrdataset() from camels_col correctly reads .nc files and returns a list of watershed ID strings."
# Uses standard variable name + sources parameter (new ADR 0001 pattern).
@needs_camels_col
@skip_if_ci
def test_read_camels_col_timeseries_xrdataset():
    ds = CamelsCol(CAMELS_COL_PATH)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["11027030"],
        var_lst=["precipitation"],  # standard name
        t_range=["1981-05-21", "1981-05-21"],
        sources={"precipitation": "observations"},  # select observations source
    )
    station_data = ts_data["precipitation"]
    result_1 = station_data.values.flatten()
    file_path = os.path.join(
        data_path,
        "CAMELS_COL",
        "04_CAMELS_COL_Hydrometeorological_data",
        "04_CAMELS_COL_Hydrometeorological_data",
        "Hydromet_data_11027030.txt.txt",
    )
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if "1981-05-21" in line:
                parts = line.split("1981-05-21", 1)[1].split()
                result_2 = parts[0].strip('"')
                break
    assert np.isclose(
        result_1[0], float(result_2), rtol=1e-6
    ), f"Expected {result_2}, got {result_1[0]}"



# "Test whether read_attr_xrdataset() from camels_se correctly reads .nc files and returns a list of watershed ID strings."
@needs_camels_se
@skip_if_ci
def test_read_camels_se_attr_xrdataset():
    ds = CamelsSe(CAMELS_SE_PATH)
    result_1 = ds.read_attr_xrdataset(
        gage_id_lst=["200"], var_lst=["urban_percentage"]
    )["urban_percentage"].values
    csv_path = os.path.join(
        data_path,
        "CAMELS_SE",
        "catchment properties",
        "catchment properties",
        "catchments_landcover.csv",
    )
    df = pd.read_csv(csv_path)
    result_2 = df[df["ID"] == 200]["Urban_percentage"].values[0]
    assert result_1 == result_2


# "Test whether read_ts_xrdataset() from camels_se correctly reads .nc files and returns a list of watershed ID strings."
# Uses standard variable name + sources parameter (new ADR 0001 pattern).
@needs_camels_se
@skip_if_ci
def test_read_camels_se_timeseries_xrdataset():
    ds = CamelsSe(CAMELS_SE_PATH)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["200"],
        var_lst=["precipitation"],  # standard name
        t_range=["1981-01-04", "1981-01-04"],
        sources={"precipitation": "default"},  # select default source
    )
    station_data = ts_data["precipitation"]
    result_1 = station_data.values.flatten()
    file_path = os.path.join(
        data_path,
        "CAMELS_SE",
        "catchment time series",
        "catchment time series",
        "catchment_id_200_RÖRVIK.csv",
    )
    csv_df = pd.read_csv(file_path)
    result_2 = csv_df.loc[
        (csv_df["Year"] == 1981) & (csv_df["Month"] == 1) & (csv_df["Day"] == 4), "Pobs_mm"
    ].values[0]
    assert np.isclose(
        result_1[0], result_2, rtol=1e-6
    ), f"Expected {result_2}, got {result_1[0]}"



# "Test whether read_attr_xrdataset() from camelsh_kr correctly reads .nc files and returns a list of watershed ID strings."
@needs_camelsh_kr
@skip_if_ci
def test_read_camelsh_kr_attr_xrdataset():
    ds = CamelshKr(CAMELSH_KR_PATH)
    result_1 = ds.read_attr_xrdataset(gage_id_lst=["1001655"], var_lst=["p_mean"])[
        "p_mean"
    ].values
    csv_path = os.path.join(
        data_path,
        "CAMELS_SK",
        "attributes_climate_ERA5Land.csv",
    )
    df = pd.read_csv(csv_path)
    result_2 = df[df["STAID"] == 1001655]["p_mean"].values[0]
    assert result_1 == result_2


# "Test whether read_ts_xrdataset() from camelsh_kr correctly reads .nc files and returns a list of watershed ID strings."
# Uses standard variable name + sources parameter (new ADR 0001 pattern).
@needs_camelsh_kr
@skip_if_ci
def test_read_camelsh_kr_timeseries_xrdataset():
    ds = CamelshKr(CAMELSH_KR_PATH)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["1001655"],
        var_lst=["precipitation"],  # standard name
        t_range=["2001-01-04", "2001-01-04"],
        sources={"precipitation": "era5"},  # select ERA5 source
    )
    station_data = ts_data["precipitation"]
    result_1 = station_data.values.flatten()[0]
    file_path = os.path.join(
        data_path,
        "CAMELS_SK",
        "timeseries",
        "timeseries",
        "1001655.csv",
    )
    csv_df = pd.read_csv(file_path)
    result_2 = csv_df[csv_df["DateTime"] == "04-Jan-2001 00:00:00"][
        "total_precipitation"
    ].values[0]
    assert np.isclose(
        result_1, result_2, rtol=1e-6
    ), f"Expected {result_2}, got {result_1}"



# "Test whether read_attr_xrdataset() from camels_gb correctly reads .nc files and returns a list of watershed ID strings."
@needs_camels_gb
@skip_if_ci
def test_read_camels_gb_attr_xrdataset():
    ds = CamelsGb(CAMELS_GB_PATH)
    result_1 = ds.read_attr_xrdataset(gage_id_lst=["102001"], var_lst=["p_mean"])[
        "p_mean"
    ].values
    csv_path = os.path.join(
        data_path,
        "CAMELS_GB",
        "camels_gb",
        "camels_gb",
        "data",
        "CAMELS_GB_climatic_attributes.csv",
    )
    df = pd.read_csv(csv_path)
    result_2 = df[df["gauge_id"] == 102001]["p_mean"].values[0]
    assert result_1 == result_2


# "Test whether read_ts_xrdataset() from camels_gb correctly reads .nc files and returns a list of watershed ID strings."
# Uses standard variable name + sources parameter (new ADR 0001 pattern).
@needs_camels_gb
@skip_if_ci
def test_read_camels_gb_timeseries_xrdataset():
    ds = CamelsGb(CAMELS_GB_PATH)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["102001"],
        var_lst=["precipitation"],  # standard name
        t_range=["1981-01-04", "1981-01-04"],
        sources={"precipitation": "meteorological"},  # select meteorological source
    )
    station_data = ts_data["precipitation"]
    result_1 = station_data.values.flatten()
    file_path = os.path.join(
        data_path,
        "CAMELS_GB",
        "camels_gb",
        "camels_gb",
        "data",
        "timeseries",
        "CAMELS_GB_hydromet_timeseries_102001_19701001-20150930.csv",
    )
    ds_csv = pd.read_csv(file_path)
    result_2 = ds_csv[ds_csv["date"] == "1981-01-04"]["precipitation"].values[0]
    assert np.isclose(
        result_1, result_2, rtol=1e-6
    ), f"Expected {result_2}, got {result_1}"


# "Test whether read_attr_xrdataset() from camels_fi correctly reads .nc files and returns a list of watershed ID strings."
@needs_camels_fi
@skip_if_ci
def test_read_camels_fi_attr_xrdataset():
    ds = CamelsFi(CAMELS_FI_PATH)
    result_1 = ds.read_attr_xrdataset(gage_id_lst=["1012"], var_lst=["p_mean"])[
        "p_mean"
    ].values
    csv_path = os.path.join(
        data_path,
        "CAMELS_FI",
        "CAMELS-FI",
        "CAMELS-FI",
        "data",
        "CAMELS_FI_climatic_attributes.csv",
    )
    df = pd.read_csv(csv_path)
    result_2 = df[df["gauge_id"] == "1012"]["p_mean"].values[0]
    assert result_1 == result_2


# "Test whether read_ts_xrdataset() from camels_fi correctly reads .nc files and returns a list of watershed ID strings."
# Uses standard variable name + sources parameter (new ADR 0001 pattern).
@needs_camels_fi
@skip_if_ci
def test_read_camels_fi_timeseries_xrdataset():
    ds = CamelsFi(CAMELS_FI_PATH)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["1012"],
        var_lst=["precipitation"],  # standard name
        t_range=["1981-01-04", "1981-01-04"],
        sources={"precipitation": "fmi"},  # select FMI source
    )
    station_data = ts_data["precipitation"]
    result_1 = station_data.values.flatten()[0]
    file_path = os.path.join(
        data_path,
        "CAMELS_FI",
        "CAMELS-FI",
        "CAMELS-FI",
        "data",
        "timeseries",
        "CAMELS_FI_hydromet_timeseries_1012_19610101-20231231.csv",
    )
    ts_df = pd.read_csv(file_path)
    result_2 = ts_df[ts_df["date"] == "1981-01-04"]["precipitation"].values[0]
    assert np.isclose(
        result_1, result_2, rtol=1e-6
    ), f"Expected {result_2}, got {result_1}"



# "Test whether read_attr_xrdataset() from camels_pe correctly reads .nc files and returns a list of watershed ID strings."
@needs_camels_pe
@skip_if_ci
def test_read_camels_pe_attr_xrdataset():
    ds = CamelsPe(CAMELS_PE_PATH)
    result_1 = ds.read_attr_xrdataset(gage_id_lst=["PE_110139"], var_lst=["p_mean"])[
        "p_mean"
    ].values
    csv_path = os.path.join(
        data_path,
        "CAMELS_PE",
        "CAMELS-PE_v1.0.1",
        "CAMELS-PE",
        "02_attributes",
        "climatic_indices.csv",
    )
    df = pd.read_csv(csv_path)
    result_2 = df[df["gauge_id"] == "PE_110139"]["p_mean"].values[0]
    assert result_1 == result_2


# "Test whether read_ts_xrdataset() from camels_pe correctly reads .nc files and returns a list of watershed ID strings."
@needs_camels_pe
@skip_if_ci
def test_read_camels_pe_timeseries_xrdataset():
    ds = CamelsPe(CAMELS_PE_PATH)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["PE_110139"],
        var_lst=["precipitation"],
        t_range=["1981-01-04", "1981-01-04"],
        sources={"precipitation": "pisco"},
    )
    result_1 = ts_data["precipitation"].values.flatten()[0]
    file_path = os.path.join(
        data_path,
        "CAMELS_PE",
        "CAMELS-PE_v1.0.1",
        "CAMELS-PE",
        "03_timeseries",
        "by_catchment",
        "PE_110139.csv",
    )
    ts_df = pd.read_csv(file_path)
    result_2 = ts_df[ts_df["date"] == "1981-01-04"]["prec"].values[0]
    assert np.isclose(
        result_1, result_2, rtol=1e-6
    ), f"Expected {result_2}, got {result_1}"


@needs_camels_pe
@skip_if_ci
def test_read_camels_pe_unified_cols_api():
    ds = CamelsPe(CAMELS_PE_PATH)
    attr = ds.read_constant_cols(
        object_ids=["PE_110139"],
        constant_cols=["area"],
    )
    ts = ds.read_relevant_cols(
        object_ids=["PE_110139"],
        t_range_list=["1981-01-04", "1981-01-04"],
        relevant_cols=["precipitation"],
    )
    assert attr.shape == (1, 1)
    assert ts.shape == (1, 1, 1)
    assert np.isfinite(attr[0, 0])
    assert np.isclose(ts[0, 0, 0], 9.783, rtol=1e-6)


def test_read_camels_pe_cloud_object_ids(monkeypatch):
    from io import StringIO

    opened = []

    class FakeFs:
        def open(self, path):
            opened.append(path)
            assert (
                path
                == "bucket/prefix/CAMELS_PE/CAMELS-PE_v1.0.1/CAMELS-PE/01_metadata/stations.csv"
            )
            return StringIO("gauge_id\nPE_2\nPE_1\n")

    monkeypatch.setattr(CamelsPe, "_make_s3fs", lambda self: FakeFs())
    ds = CamelsPe("s3://bucket/prefix")
    assert ds.read_object_ids().tolist() == ["PE_1", "PE_2"]
    assert opened == [
        "bucket/prefix/CAMELS_PE/CAMELS-PE_v1.0.1/CAMELS-PE/01_metadata/stations.csv"
    ]


# "Test whether read_attr_xrdataset() from camels_lux correctly reads .nc files and returns a list of watershed ID strings."
@needs_camels_lux
@skip_if_ci
def test_read_camels_lux_attr_xrdataset():
    ds = CamelsLux(CAMELS_LUX_PATH)
    result_1 = ds.read_attr_xrdataset(gage_id_lst=["ID_05"], var_lst=["Qspec_sum"])[
        "Qspec_sum"
    ].values
    csv_path = os.path.join(
        data_path,
        "CAMELS_LUX",
        "CAMELS-LUX",
        "CAMELS_LUX_climatic_attributes.csv",
    )
    df = pd.read_csv(csv_path)
    result_2 = df[df["gauge_id"] == "ID_05"]["Qspec_sum"].values[0]
    assert result_1 == result_2


# "Test whether read_ts_xrdataset() from camels_lux correctly reads .nc files and returns a list of watershed ID strings."
# Uses standard variable name + sources parameter (new ADR 0001 pattern).
@needs_camels_lux
@skip_if_ci
def test_read_camels_lux_timeseries_xrdataset():
    ds = CamelsLux(CAMELS_LUX_PATH)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["ID_05"],
        var_lst=["streamflow"],  # standard name
        t_range=["2005-01-04", "2005-01-04"],
        sources={"streamflow": "observations"},  # select observations source
    )
    station_data = ts_data["streamflow"]
    result_1 = station_data.values.flatten()
    file_path = os.path.join(
        data_path,
        "CAMELS_LUX",
        "CAMELS-LUX",
        "timeseries",
        "daily",
        "CAMELS_LUX_hydromet_timeseries_ID_05.csv",
    )
    ds_csv = pd.read_csv(file_path)
    result_2 = ds_csv[ds_csv["Date"] == "2005-01-04"]["Q"].values[0]
    assert np.isclose(
        result_1, result_2, rtol=1e-6
    ), f"Expected {result_2}, got {result_1}"



# "Test whether read_attr_xrdataset() from camels_nz correctly reads .nc files and returns a list of watershed ID strings."
@needs_camels_nz
@skip_if_ci
def test_read_camels_nz_attr_xrdataset():
    ds = CamelsNz(CAMELS_NZ_PATH)
    result_1 = ds.read_attr_xrdataset(gage_id_lst=["3819"], var_lst=["area"])[
        "area"
    ].values
    csv_path = os.path.join(
        data_path,
        "CAMELS_NZ",
        "camels_nz",
        "CAMELS_NZ_Catchment_Atrributes",
        "1.CAMELS_NZ_Catchment_information.csv",
    )
    df = pd.read_csv(csv_path)

    result_2 = df[df["Station_ID"] == 3819]["uparea"].values[0]
    assert result_1 == result_2


# "Test whether read_ts_xrdataset() from camels_nz correctly reads .nc files and returns a list of watershed ID strings."
# Uses standard variable name + sources parameter (new ADR 0001 pattern).
@needs_camels_nz
@skip_if_ci
def test_read_camels_nz_timeseries_xrdataset():
    ds = CamelsNz(CAMELS_NZ_PATH)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["3819"],
        var_lst=["streamflow"],  # standard name
        t_range=["1981-01-04", "1981-01-04"],
        sources={"streamflow": "obs"},  # select observations source
    )
    station_data = ts_data["streamflow"]
    result_1 = station_data.values.flatten()[0]
    file_path = os.path.join(
        data_path,
        "CAMELS_NZ",
        "camels_nz",
        "CAMELS_NZ_hourly_Streamflow",
        "flow_station_id_3819.csv",
    )
    ds = pd.read_csv(file_path)
    result_2 = ds[ds["time"] == "1981-01-04 00:00:00"]["flow"].values[0]
    assert np.isclose(
        result_1, result_2, rtol=1e-6
    ), f"Expected {result_2}, got {result_1}"



# "Test whether read_attr_xrdataset() from camels_de correctly reads .nc files and returns a list of watershed ID strings."
@needs_camels_de
@skip_if_ci
def test_read_camels_de_attr_xrdataset():
    ds = CamelsDe(CAMELS_DE_PATH)
    result_1 = ds.read_attr_xrdataset(gage_id_lst=["DE110010"], var_lst=["p_mean"])[
        "p_mean"
    ].values
    csv_path = os.path.join(
        data_path,
        "CAMELS_DE",
        "camels_de",
        "CAMELS_DE_climatic_attributes.csv",
    )
    df = pd.read_csv(csv_path)
    result_2 = df[df["gauge_id"] == "DE110010"]["p_mean"].values[0]
    assert result_1 == result_2


# "Test whether read_ts_xrdataset() from camels_de correctly reads .nc files and returns a list of watershed ID strings."
# Uses standard variable name + sources parameter (new ADR 0001 pattern).
@needs_camels_de
@skip_if_ci
def test_read_camels_de_timeseries_xrdataset():
    ds = CamelsDe(CAMELS_DE_PATH)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["DE110010"],
        var_lst=["streamflow"],  # standard name
        t_range=["1981-01-04", "1981-01-04"],
        sources={"streamflow": "vol"},  # the default source (volumetric discharge)
    )
    station_data = ts_data["streamflow"]
    result_1 = station_data.values.flatten()
    file_path = os.path.join(
        data_path,
        "CAMELS_DE",
        "camels_de",
        "timeseries",
        "CAMELS_DE_hydromet_timeseries_DE110010.csv",
    )
    csv_df = pd.read_csv(file_path)
    result_2 = csv_df[csv_df["date"] == "1981-01-04"]["discharge_vol_obs"].values[0]
    assert np.isclose(
        result_1, result_2, rtol=1e-6
    ), f"Expected {result_2}, got {result_1}"



# "Test whether read_attr_xrdataset() from camels_fr correctly reads .nc files and returns a list of watershed ID strings."
@needs_camels_fr
@skip_if_ci
def test_read_camels_fr_attr_xrdataset():
    ds = CamelsFr(CAMELS_FR_PATH)
    result_1 = ds.read_attr_xrdataset(
        gage_id_lst=["A140202001"], var_lst=["hgl_krs_not_karstic"]
    )["hgl_krs_not_karstic"].values
    csv_path = os.path.join(
        data_path,
        "CAMELS_FR",
        "CAMELS_FR_attributes",
        "CAMELS_FR_attributes",
        "static_attributes",
        "CAMELS_FR_hydrogeology_attributes.csv",
    )
    df = pd.read_csv(csv_path, sep=";", skiprows=0, header=0)

    result_2 = df[df["sta_code_h3"] == "A140202001"]["hgl_krs_not_karstic"].values[0]
    assert result_1 == result_2


# "Test whether read_ts_xrdataset() from camels_fr correctly reads .nc files and returns a list of watershed ID strings."
# Uses standard variable name + sources parameter (new ADR 0001 pattern).
@needs_camels_fr
@skip_if_ci
def test_read_camels_fr_timeseries_xrdataset():
    ds = CamelsFr(CAMELS_FR_PATH)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["A140202001"],
        var_lst=["streamflow"],  # standard name
        t_range=["1981-01-04", "1981-01-04"],
        sources={"streamflow": "hydroportail"},  # select hydroportail source (default)
    )
    station_data = ts_data["streamflow"]
    result_1 = station_data.values.flatten()
    file_path = os.path.join(
        data_path,
        "CAMELS_FR",
        "CAMELS_FR_time_series",
        "CAMELS_FR_time_series",
        "daily",
        "CAMELS_FR_tsd_A140202001.csv",
    )
    df = pd.read_csv(file_path, sep=";", skiprows=7, header=0)
    result_2 = df[df["tsd_date"] == 19810104]["tsd_q_l"].values[0]
    assert np.isclose(
        result_1, result_2, rtol=1e-6
    ), f"Expected {result_2}, got {result_1}"


# "Test whether read_attr_xrdataset() from camels_ch correctly reads .nc files and returns a list of watershed ID strings."
@needs_camels_ch
@skip_if_ci
def test_read_camels_ch_attr_xrdataset():
    ds = CamelsCh(CAMELS_CH_PATH)
    result_1 = ds.read_attr_xrdataset(gage_id_lst=["2109"], var_lst=["p_mean"])[
        "p_mean"
    ].values
    csv_path = os.path.join(
        data_path,
        "CAMELS_CH",
        "camels_ch",
        "camels_ch",
        "static_attributes",
        "CAMELS_CH_climate_attributes_obs.csv",
    )
    df = pd.read_csv(csv_path, sep=",", skiprows=1, header=0)

    result_2 = df[df["gauge_id"] == 2109]["p_mean"].values[0]
    assert result_1 == result_2


# "Test whether read_ts_xrdataset() from camels_ch correctly reads .nc files and returns a list of watershed ID strings."
# Uses standard variable name + sources parameter (new ADR 0001 pattern).
@needs_camels_ch
@skip_if_ci
def test_read_camels_ch_timeseries_xrdataset():
    ds = CamelsCh(CAMELS_CH_PATH)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["2109"],
        var_lst=["precipitation"],  # standard name
        t_range=["1981-01-04", "1981-01-04"],
        sources={"precipitation": "sfo"},  # select SFO source
    )
    station_data = ts_data["precipitation"]
    result_1 = station_data.values.flatten()
    file_path = os.path.join(
        data_path,
        "CAMELS_CH",
        "camels_ch",
        "camels_ch",
        "timeseries",
        "observation_based",
        "CAMELS_CH_obs_based_2109.csv",
    )
    ds_csv = pd.read_csv(file_path, sep=",", header=0)
    result_2 = ds_csv[ds_csv["date"] == "1981-01-04"]["precipitation(mm/d)"].values[0]
    assert np.isclose(
        result_1, result_2, rtol=1e-6
    ), f"Expected {result_2}, got {result_1}"


# "Test whether read_attr_xrdataset() from camels_ind correctly reads static attrs."
@needs_camels_ind
@skip_if_ci
def test_read_camels_ind_attr_xrdataset():
    ds = CamelsInd(CAMELS_IND_PATH)
    result_1 = ds.read_attr_xrdataset(gage_id_lst=["10001"], var_lst=["p_mean"])[
        "p_mean"
    ].values
    # Ground-truth comparison from raw CSV (semicolon-separated)
    csv_path = os.path.join(
        data_path,
        "CAMELS_IND",
        "CAMELS_IND_All_Catchments",
        "attributes_txt",
        "camels_ind_clim.txt",
    )
    df = pd.read_csv(csv_path, sep=";")
    result_2 = df[df["gauge_id"].astype(str) == "10001"]["p_mean"].values[0]
    assert np.isclose(
        result_1, result_2, rtol=1e-6
    ), f"Expected {result_2}, got {result_1}"


# Uses standard variable name + sources parameter (new ADR 0001 pattern).
@needs_camels_ind
@skip_if_ci
def test_read_camels_ind_timeseries_xrdataset():
    ds = CamelsInd(CAMELS_IND_PATH)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["10001"],
        var_lst=["precipitation"],  # standard name
        t_range=["1990-06-01", "1990-06-01"],
        sources={"precipitation": "imd"},  # IMD precipitation source (default)
    )
    result_1 = ts_data["precipitation"].values.flatten()[0]
    # Ground-truth comparison from raw forcing CSV
    file_path = os.path.join(
        data_path,
        "CAMELS_IND",
        "CAMELS_IND_All_Catchments",
        "catchment_mean_forcings",
        "10001.csv",
    )
    df = pd.read_csv(file_path)
    result_2 = df[(df["year"] == 1990) & (df["month"] == 6) & (df["day"] == 1)][
        "prcp(mm/day)"
    ].values[0]
    assert np.isclose(
        result_1, result_2, rtol=1e-6
    ), f"Expected {result_2}, got {result_1}"


# "Test whether read_attr_xrdataset() from camels_br correctly reads static attrs."
@needs_camels_br
@skip_if_ci
def test_read_camels_br_attr_xrdataset():
    ds = CamelsBr(CAMELS_BR_PATH)
    result_1 = ds.read_attr_xrdataset(gage_id_lst=["10500000"], var_lst=["area"])[
        "area"
    ].values
    # Ground-truth comparison from raw CSV (whitespace-separated)
    csv_path = os.path.join(
        data_path,
        "CAMELS_BR",
        "01_CAMELS_BR_attributes",
        "01_CAMELS_BR_attributes",
        "camels_br_topography.txt",
    )
    df = pd.read_csv(csv_path, sep=r"\s+")
    result_2 = df[df["gauge_id"].astype(str) == "10500000"]["area"].values[0]
    assert np.isclose(
        result_1, result_2, rtol=1e-6
    ), f"Expected {result_2}, got {result_1}"


# Uses standard variable name + sources parameter (new ADR 0001 pattern).
@needs_camels_br
@skip_if_ci
def test_read_camels_br_timeseries_xrdataset():
    ds = CamelsBr(CAMELS_BR_PATH)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["10500000"],
        var_lst=["streamflow"],  # standard name
        t_range=["2000-06-01", "2000-06-01"],
        sources={"streamflow": "m3s"},  # observed streamflow in m^3/s (default)
    )
    result_1 = ts_data["streamflow"].values.flatten()[0]
    # Ground-truth comparison from raw streamflow file
    file_path = os.path.join(
        data_path,
        "CAMELS_BR",
        "03_CAMELS_BR_streamflow_selected_catchments",
        "03_CAMELS_BR_streamflow_selected_catchments",
        "10500000_streamflow.txt",
    )
    df = pd.read_csv(file_path, sep=r"\s+")
    result_2 = df[(df["year"] == 2000) & (df["month"] == 6) & (df["day"] == 1)][
        "streamflow_m3s"
    ].values[0]
    assert np.isclose(
        result_1, result_2, rtol=1e-6
    ), f"Expected {result_2}, got {result_1}"


# ═══════════════════════════════════════════════════════════════════════
# CamelsUs-specific fixtures and helpers
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_aqua_camels():
    """Patch CAMELS_US from aqua_fetch so unit tests don't hit network."""
    with patch("hydrodataset.camels_us.CAMELS_US", autospec=True) as mock:
        mock.return_value.dynamic_features = []
        mock.return_value.static_features = []
        yield mock


needs_camels_us_data = pytest.mark.skipif(
    CAMELS_US_PATH is None or bool(os.getenv("CI")),
    reason="CAMELS_US data not available or CI (large dataset download)",
)


def _require_camels_us_data():
    """Skip if CAMELS_US data is not available."""
    if CAMELS_US_PATH is None:
        pytest.skip("CAMELS_US data not available — configure ~/hydro_setting.yml")


# ═══════════════════════════════════════════════════════════════════
# CamelsUs unit tests — no real data needed
# ═══════════════════════════════════════════════════════════════════


class TestResolver:
    """Tests for camels_us in the data resolver / registry."""

    def test_camels_us_in_reader_aliases(self):
        """camels_us must be a registered reader alias."""
        from hydrodataset.configs.data_resolver import READER_ALIASES

        assert "camels_us" in READER_ALIASES
        spec = READER_ALIASES["camels_us"]
        assert spec["class"] == "CamelsUs"
        assert spec["module"] == "hydrodataset.camels_us"
        assert spec["category"] == "hydrodataset"

    def test_resolve_returns_absolute_path(self):
        """resolve_data_path should return an absolute path pointing to existing directory."""
        _require_camels_us_data()
        path = resolve_data_path("camels_us")
        assert os.path.isabs(path), f"Expected absolute path, got: {path}"
        assert os.path.isdir(path), f"Path does not exist: {path}"


class TestCamelsUsConstruction:
    """Tests for CamelsUs.__init__ behavior.

    All tests use mock_aqua_camels to avoid triggering aqua_fetch network calls
    during construction with an empty tmp_path.
    """

    def test_constructor_accepts_absolute_path(self, tmp_path, mock_aqua_camels):
        """Constructor must accept an absolute path and set data_source_dir."""
        ds = CamelsUs(str(tmp_path))
        assert ds.data_source_dir == tmp_path

    def test_constructor_rejects_relative_path(self, mock_aqua_camels):
        """Constructor must reject relative paths with a helpful error message."""
        with pytest.raises(ValueError, match="uri must be an absolute path"):
            CamelsUs("relative/path/to/data")

    def test_constructor_accepts_s3_uri(self, mock_aqua_camels, tmp_path):
        """Constructor must accept S3 URIs without Path validation."""
        s3_uri = "s3://test-bucket/prefix/camels_us"
        ds = CamelsUs(s3_uri)
        assert ds.data_source_dir == s3_uri

    def test_constructor_creates_dir_if_missing(self, tmp_path, mock_aqua_camels):
        """Constructor should create data_source_dir if it does not exist."""
        new_dir = tmp_path / "new_camels_us_dir"
        assert not new_dir.exists()
        ds = CamelsUs(str(new_dir))
        try:
            assert ds.data_source_dir == new_dir
            assert new_dir.exists()
        finally:
            if new_dir.exists():
                new_dir.rmdir()

    def test_default_region_is_us(self, tmp_path, mock_aqua_camels):
        """Default region should be 'US'."""
        ds = CamelsUs(str(tmp_path))
        assert ds.region == "US"

    def test_explicit_region(self, tmp_path, mock_aqua_camels):
        """Region should be settable via the region parameter."""
        ds = CamelsUs(str(tmp_path), region="US")
        assert ds.region == "US"


class TestCamelsUsMetadata:
    """Tests for CamelsUs class-level metadata (no data needed)."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, mock_aqua_camels):
        self.ds = CamelsUs(str(tmp_path))

    def test_attributes_cache_filename(self):
        """Attribute cache filename should be set."""
        assert self.ds._attributes_cache_filename == "camels_us_attributes.nc"

    def test_timeseries_cache_filename(self):
        """Timeseries cache filename should be set."""
        assert self.ds._timeseries_cache_filename == "camels_us_timeseries.nc"

    def test_default_t_range(self):
        """Default time range should cover the CAMELS-US observation period."""
        t_range = self.ds.default_t_range
        assert t_range == ["1980-01-01", "2014-12-31"]

    def test_static_definitions_has_expected_keys(self):
        """Merged static definitions should contain both base and subclass attributes."""
        defs = self.ds._static_variable_definitions
        # Base class attributes
        assert "area" in defs
        assert "p_mean" in defs
        # Subclass attributes
        assert "elev_mean" in defs
        assert "slope_mean" in defs
        assert "gauge_lat" in defs
        assert "gauge_lon" in defs

    def test_subclass_static_definitions_has_expected_keys(self):
        """Subclass-specific static definitions should contain CAMELS-US attributes."""
        defs = self.ds._subclass_static_definitions
        assert "elev_mean" in defs
        assert "slope_mean" in defs
        assert "gauge_lat" in defs
        assert "gauge_lon" in defs
        assert "huc_02" in defs  # Unique to CAMELS-US

    def test_subclass_static_definitions_structure(self):
        """Each static definition must have specific_name and unit."""
        for var_name, spec in self.ds._subclass_static_definitions.items():
            assert "specific_name" in spec, f"{var_name}: missing specific_name"
            assert "unit" in spec, f"{var_name}: missing unit"

    def test_dynamic_variable_mapping_has_expected_keys(self):
        """Dynamic variable mapping should contain standard meteorological variables."""
        from hydrodataset import StandardVariable

        mapping = self.ds._dynamic_variable_mapping
        required = [
            StandardVariable.STREAMFLOW,
            StandardVariable.PRECIPITATION,
            StandardVariable.TEMPERATURE_MAX,
            StandardVariable.TEMPERATURE_MIN,
        ]
        for v in required:
            assert v in mapping, f"Missing required dynamic variable: {v}"

    def test_dynamic_variable_sources_structure(self):
        """Each dynamic variable should have default_source and sources dict."""
        for std_var, spec in self.ds._dynamic_variable_mapping.items():
            assert "default_source" in spec, f"{std_var}: missing default_source"
            assert "sources" in spec, f"{std_var}: missing sources"
            default_src = spec["default_source"]
            assert default_src in spec["sources"], (
                f"{std_var}: default_source '{default_src}' not in sources"
            )
            for src_name, src_spec in spec["sources"].items():
                assert "specific_name" in src_spec, (
                    f"{std_var}/{src_name}: missing specific_name"
                )
                assert "unit" in src_spec, f"{std_var}/{src_name}: missing unit"

    def test_pet_variable_in_mapping(self):
        """PET and ET should be available as dynamic variables."""
        from hydrodataset import StandardVariable

        mapping = self.ds._dynamic_variable_mapping
        assert StandardVariable.POTENTIAL_EVAPOTRANSPIRATION in mapping
        assert StandardVariable.EVAPOTRANSPIRATION in mapping


class TestCamelsUsDynamicFeatures:
    """Tests for _dynamic_features (PET override)."""

    def test_dynamic_features_includes_pet_and_et(self, tmp_path, mock_aqua_camels):
        """_dynamic_features should include PET and ET from the override."""
        ds = CamelsUs(str(tmp_path))
        features = ds._dynamic_features()
        assert "PET" in features
        assert "ET" in features


# ═══════════════════════════════════════════════════════════════════
# Integration tests — require CAMELS_US dataset on disk
# ═══════════════════════════════════════════════════════════════════


KNOWN_GAUGE = "01013500"  # St. John River at Ninemile Bridge, Maine
# Known values from CAMELS-US v1.2 camels_topo.txt (verified against aqua_fetch cache):
# area_km2 = 2252.7 (gauge_dam / gauge_area; differs from area_gages2 = 687.52)
KNOWN_AREA_KM2 = 2252.7
KNOWN_ELEV_MEAN_M = 250.31


@needs_camels_us_data
class TestCamelsUsReadObjectIds:
    """Tests for read_object_ids() — the simplest integration smoke test."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = CamelsUs(CAMELS_US_PATH)

    def test_read_object_ids_returns_list(self):
        """Should return a non-empty list of gauge ID strings."""
        ids = self.ds.read_object_ids()
        assert isinstance(ids, (list, np.ndarray))
        assert len(ids) > 0
        assert all(isinstance(gid, str) for gid in ids[:5])

    def test_read_object_ids_contains_known_gauge(self):
        """Should contain the well-known gauge '01013500'."""
        ids = self.ds.read_object_ids()
        ids_list = [str(g) for g in ids]
        assert "01013500" in ids_list


@needs_camels_us_data
class TestCamelsUsReadAttr:
    """Integration tests for read_attr_xrdataset with real data."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = CamelsUs(CAMELS_US_PATH)

    def test_read_area_for_known_gauge(self):
        """read_attr_xrdataset should return the known area for gauge 01013500."""
        result = self.ds.read_attr_xrdataset(
            gage_id_lst=[KNOWN_GAUGE], var_lst=["area"]
        )
        area = result["area"].values
        assert len(area) == 1
        assert np.isclose(area[0], KNOWN_AREA_KM2, rtol=0.01), (
            f"Expected area={KNOWN_AREA_KM2}, got {area[0]}. "
            f"If data changed, delete cache and regenerate."
        )

    def test_read_elev_mean_for_known_gauge(self):
        """read_attr_xrdataset should return the known elevation for gauge 01013500."""
        result = self.ds.read_attr_xrdataset(
            gage_id_lst=[KNOWN_GAUGE], var_lst=["elev_mean"]
        )
        elev = result["elev_mean"].values
        assert len(elev) == 1
        assert np.isclose(elev[0], KNOWN_ELEV_MEAN_M, rtol=0.01), (
            f"Expected elev_mean={KNOWN_ELEV_MEAN_M}, got {elev[0]}"
        )

    def test_read_multiple_vars(self):
        """Should be able to read multiple attributes at once."""
        result = self.ds.read_attr_xrdataset(
            gage_id_lst=[KNOWN_GAUGE], var_lst=["area", "elev_mean", "p_mean"]
        )
        for var in ["area", "elev_mean", "p_mean"]:
            assert var in result.data_vars, f"Missing variable: {var}"

    def test_read_multiple_gauges(self):
        """Should be able to read attributes for multiple gauges at once."""
        gauges = [KNOWN_GAUGE, "01022500"]
        result = self.ds.read_attr_xrdataset(
            gage_id_lst=gauges, var_lst=["area"]
        )
        assert "area" in result.data_vars
        assert len(result.basin) == 2
        area_values = result["area"].values
        assert area_values[0] > 0
        assert area_values[1] > 0

    def test_attributes_are_not_nan(self):
        """Verify key attributes are not NaN for the known gauge."""
        attrs_to_check = [
            "area", "elev_mean", "slope_mean", "p_mean",
            "gauge_lat", "gauge_lon",
        ]
        result = self.ds.read_attr_xrdataset(
            gage_id_lst=[KNOWN_GAUGE], var_lst=attrs_to_check
        )
        for attr in attrs_to_check:
            assert attr in result.data_vars, f"Missing: {attr}"
            val = result[attr].values[0]
            assert not np.isnan(val), f"{attr} is NaN for {KNOWN_GAUGE}"


@needs_camels_us_data
class TestCamelsUsReadTs:
    """Integration tests for read_ts_xrdataset with real data."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = CamelsUs(CAMELS_US_PATH)

    def test_read_streamflow_single_day(self):
        """Should read observed streamflow for a known gauge and single day."""
        ts_data = self.ds.read_ts_xrdataset(
            gage_id_lst=[KNOWN_GAUGE],
            var_lst=["streamflow"],
            t_range=["1981-01-04", "1981-01-04"],
            sources={"streamflow": "usgs"},
        )
        assert "streamflow" in ts_data.data_vars
        values = ts_data["streamflow"].values
        assert values.shape == (1, 1)
        val = values[0, 0]
        assert val >= 0, f"Streamflow should be non-negative, got {val}"
        assert val < 10000, f"Streamflow unreasonably large: {val}"

    def test_read_precipitation_single_day(self):
        """Should read precipitation for a known gauge and single day."""
        ts_data = self.ds.read_ts_xrdataset(
            gage_id_lst=[KNOWN_GAUGE],
            var_lst=["precipitation"],
            t_range=["1981-01-04", "1981-01-04"],
            sources={"precipitation": "daymet"},
        )
        assert "precipitation" in ts_data.data_vars
        values = ts_data["precipitation"].values
        assert values.shape == (1, 1)
        val = values[0, 0]
        assert val >= 0, f"Precipitation should be non-negative, got {val}"

    def test_read_multiple_variables(self):
        """Should read multiple time-series variables at once."""
        ts_data = self.ds.read_ts_xrdataset(
            gage_id_lst=[KNOWN_GAUGE],
            var_lst=["streamflow", "precipitation", "temperature_max", "temperature_min"],
            t_range=["1981-01-01", "1981-01-03"],
            sources={
                "streamflow": "usgs",
                "precipitation": "daymet",
                "temperature_max": "daymet",
                "temperature_min": "daymet",
            },
        )
        for var in ["streamflow", "precipitation", "temperature_max", "temperature_min"]:
            assert var in ts_data.data_vars, f"Missing: {var}"
            assert ts_data[var].values.shape == (1, 3)

    def test_read_streamflow_multi_day(self):
        """Should return correct number of time steps for a multi-day range."""
        ts_data = self.ds.read_ts_xrdataset(
            gage_id_lst=[KNOWN_GAUGE],
            var_lst=["streamflow"],
            t_range=["1981-01-01", "1981-01-10"],
            sources={"streamflow": "usgs"},
        )
        assert ts_data["streamflow"].values.shape == (1, 10)

    def test_read_default_t_range(self):
        """Should work with the default_t_range (long time series)."""
        ts_data = self.ds.read_ts_xrdataset(
            gage_id_lst=[KNOWN_GAUGE],
            var_lst=["streamflow"],
            sources={"streamflow": "usgs"},
        )
        n_days = ts_data["streamflow"].values.shape[1]
        assert n_days > 10000, f"Expected >10000 days, got {n_days}"


@needs_camels_us_data
class TestCamelsUsModelOutput:
    """Integration tests for read_camels_us_model_output_data (PET/ET)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = CamelsUs(CAMELS_US_PATH)

    def test_read_pet_single_day(self):
        """Should read model-output PET for a known gauge."""
        data = self.ds.read_camels_us_model_output_data(
            gage_id_lst=[KNOWN_GAUGE],
            t_range=["1981-01-04", "1981-01-04"],
            var_lst=["PET"],
        )
        assert isinstance(data, np.ndarray)
        assert data.shape == (1, 1, 1)
        val = data[0, 0, 0]
        assert val >= 0 or np.isnan(val), (
            f"PET should be non-negative (or NaN if missing), got {val}"
        )

    def test_read_pet_et_together(self):
        """Should read both PET and ET in one call."""
        data = self.ds.read_camels_us_model_output_data(
            gage_id_lst=[KNOWN_GAUGE],
            t_range=["1981-01-04", "1981-01-04"],
            var_lst=["PET", "ET"],
        )
        assert data.shape == (1, 1, 2)

    def test_read_pet_multi_day(self):
        """Should return correct shape for a multi-day range."""
        data = self.ds.read_camels_us_model_output_data(
            gage_id_lst=[KNOWN_GAUGE],
            t_range=["1981-01-01", "1981-01-10"],
            var_lst=["PET"],
        )
        assert data.shape[1] == 10


@needs_camels_us_data
class TestCamelsUsCache:
    """Tests that verify cache files are created and readable."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = CamelsUs(CAMELS_US_PATH)

    def test_attributes_cache_exists(self):
        """Attribute cache file should exist after first read."""
        self.ds.read_attr_xrdataset(gage_id_lst=[KNOWN_GAUGE], var_lst=["area"])
        cache_file = self.ds.cache_dir.joinpath(
            self.ds._attributes_cache_filename
        )
        assert cache_file.exists(), (
            f"Attribute cache not found at: {cache_file}"
        )

    def test_timeseries_cache_exists(self):
        """Timeseries cache file should exist after first read."""
        self.ds.read_ts_xrdataset(
            gage_id_lst=[KNOWN_GAUGE],
            var_lst=["streamflow"],
            t_range=["1981-01-01", "1981-01-01"],
            sources={"streamflow": "usgs"},
        )
        cache_file = self.ds.cache_dir.joinpath(
            self.ds._timeseries_cache_filename
        )
        assert cache_file.exists(), (
            f"Timeseries cache not found at: {cache_file}"
        )
