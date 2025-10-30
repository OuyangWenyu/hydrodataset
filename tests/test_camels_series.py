import xarray as xr
from hydrodataset import CACHE_DIR, SETTING
import numpy as np
import pandas as pd
import os
import pytest
from hydrodataset.camelsh import Camelsh  # 从 camelsh.py 导入
from hydrodataset.camels_aus import CamelsAus  # 从 camels_aus_aqua.py 导入
from hydrodataset.camels_cl import CamelsCl  # 从 camels_cl_aqua.py 导入
from hydrodataset.camels_dk import CamelsDk  # 从 camels_dk_aqua.py 导入
from hydrodataset.camels_col import CamelsCol  # 从 camels_col_aqua.py 导入
from hydrodataset.camels_se import CamelsSe  # 从 camels_se_aqua.py 导入
from hydrodataset.camelsh_kr import CamelshKr  # 从 camelsh_kr.py 导入
from hydrodataset.camels_gb import CamelsGb  # 从 camels_gb_aqua.py 导入
from hydrodataset.camels_fi import CamelsFi  # 从 camels_fi_aqua.py 导入
from hydrodataset.camels_lux import CamelsLux  # 从 camels_lux_aqua.py 导入
from hydrodataset.camels_nz import CamelsNz  # 从 camels_nz_aqua.py 导入
from hydrodataset.camels_de import CamelsDe  # 从 camels_de_aqua.py 导入
from hydrodataset.camels_fr import CamelsFr  # 从 camels_fr_aqua.py 导入
from hydrodataset.camels_ch import CamelsCh  # 从 camels_ch_aqua.py 导入

data_path = SETTING["local_data_path"]["datasets-origin"]


# "Test whether read_attr_xrdataset() from camelsh correctly reads .nc files and returns a list of watershed ID strings."
@pytest.mark.skip(reason="Requires large dataset download, not suitable for CI")
def test_read_attr_xrdataset():
    ds = Camelsh(data_path)
    result_1 = ds.read_attr_xrdataset(gage_id_lst="01011000", var_lst=["p_mean"])[
        "p_mean"
    ].values
    csv_path = os.path.join(
        data_path,
        "CAMELSH",
        "attributes",
        "attributes_nldas2_climate.csv",
    )
    df = pd.read_csv(csv_path)
    result_2 = df[df["STAID"] == 1011000]["p_mean"].values[0]
    assert result_1 == result_2


# "Test whether read_ts_xrdataset() from camelsh correctly reads .nc files and returns a list of watershed ID strings."
@pytest.mark.skip(reason="Requires large dataset download, not suitable for CI")
def test_read_timeseries_xrdataset():
    ds = Camelsh(data_path)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["01011000"],
        var_lst=["pet_mm"],
        t_range=["1980-01-01", "1980-01-01"],
    )
    station_data = ts_data["pet_mm"]
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
    values_match = np.array_equal(result_1, result_2)
    assert values_match


# "Test whether read_attr_xrdataset() from camels_aus correctly reads .nc files and returns a list of watershed ID strings."
@pytest.mark.skip(reason="Requires large dataset download, not suitable for CI")
def test_read_camels_aus_attr_xrdataset():
    ds = CamelsAus(data_path)
    result_1 = ds.read_attr_xrdataset(gage_id_lst="912105A", var_lst=["anngro_mega"])[
        "anngro_mega"
    ].values
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
@pytest.mark.skip(reason="Requires large dataset download, not suitable for CI")
def test_read_camels_aus_timeseries_xrdataset():
    ds = CamelsAus(data_path)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["912105A"],
        var_lst=["airtemp_C_silo_min"],
        t_range=["1980-01-04", "1980-01-04"],
    )
    station_data = ts_data["airtemp_C_silo_min"]
    result_1 = station_data.values.flatten()
    file_path = os.path.join(
        data_path,
        "CAMELS_AUS",
        "05_hydrometeorology",
        "05_hydrometeorology",
        "03_Other",
        "SILO",
        "tmin_SILO.csv",
    )
    ds = pd.read_csv(file_path)
    result_2 = ds.loc[
        (ds["year"] == 1980) & (ds["month"] == 1) & (ds["day"] == 4), "912105A"
    ].values[0]
    assert result_1 == result_2


# "Test whether read_attr_xrdataset() from camels_cl correctly reads .nc files and returns a list of watershed ID strings."
@pytest.mark.skip(reason="Requires large dataset download, not suitable for CI")
def test_read_camels_cl_attr_xrdataset():
    ds = CamelsCl(data_path)
    result_1 = ds.read_attr_xrdataset(gage_id_lst="1021001", var_lst=["elev_mean"])[
        "elev_mean"
    ].values
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
@pytest.mark.skip(reason="Requires large dataset download, not suitable for CI")
def test_read_camels_cl_timeseries_xrdataset():
    ds = CamelsCl(data_path)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["1021001"],
        var_lst=["q_cms_obs"],
        t_range=["1981-01-04", "1981-01-04"],
    )
    station_data = ts_data["q_cms_obs"]
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
    assert float(result_1) == float(result_2)


# "Test whether read_attr_xrdataset() from camels_dk correctly reads .nc files and returns a list of watershed ID strings."
@pytest.mark.skip(reason="Requires large dataset download, not suitable for CI")
def test_read_camels_dk_attr_xrdataset():
    ds = CamelsDk(data_path)
    result_1 = ds.read_attr_xrdataset(gage_id_lst="12431077", var_lst=["p_mean"])[
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
@pytest.mark.skip(reason="Requires large dataset download, not suitable for CI")
def test_read_camels_dk_timeseries_xrdataset():
    ds = CamelsDk(data_path)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["12431077"],
        var_lst=["pcp_mm"],
        t_range=["1990-01-05", "1990-01-05"],
    )
    station_data = ts_data["pcp_mm"]
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
    assert result_1 == result_2


# "Test whether read_attr_xrdataset() from camels_col correctly reads .nc files and returns a list of watershed ID strings."
@pytest.mark.skip(reason="Requires large dataset download, not suitable for CI")
def test_read_camels_col_attr_xrdataset():
    ds = CamelsCol(data_path)
    result_1 = ds.read_attr_xrdataset(gage_id_lst="11027030", var_lst=["q_mean"])[
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
@pytest.mark.skip(reason="Requires large dataset download, not suitable for CI")
def test_read_camels_col_timeseries_xrdataset():
    ds = CamelsCol(data_path)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["11027030"],
        var_lst=["pcp_mm"],
        t_range=["1981-01-04", "1981-01-04"],
    )
    station_data = ts_data["pcp_mm"]
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
            if "1981-01-04" in line:
                parts = line.split("1981-01-04", 1)[1].split()
                result_2 = parts[0].strip('"')
                break
    assert float(result_1) == float(result_2)


# "Test whether read_attr_xrdataset() from camels_se correctly reads .nc files and returns a list of watershed ID strings."
@pytest.mark.skip(reason="Requires large dataset download, not suitable for CI")
def test_read_camels_se_attr_xrdataset():
    ds = CamelsSe(data_path)
    result_1 = ds.read_attr_xrdataset(gage_id_lst="200", var_lst=["Urban_percentage"])[
        "Urban_percentage"
    ].values
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
@pytest.mark.skip(reason="Requires large dataset download, not suitable for CI")
def test_read_camels_se_timeseries_xrdataset():
    ds = CamelsSe(data_path)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["200"],
        var_lst=["pcp_mm"],
        t_range=["1981-01-04", "1981-01-04"],
    )
    station_data = ts_data["pcp_mm"]
    result_1 = station_data.values.flatten()
    file_path = os.path.join(
        data_path,
        "CAMELS_SE",
        "catchment time series",
        "catchment time series",
        "catchment_id_200_RÖRVIK.csv",
    )
    ds = pd.read_csv(file_path)
    result_2 = ds.loc[
        (ds["Year"] == 1981) & (ds["Month"] == 1) & (ds["Day"] == 4), "Pobs_mm"
    ].values[0]
    assert result_1 == result_2


# "Test whether read_attr_xrdataset() from camels_sk correctly reads .nc files and returns a list of watershed ID strings."
@pytest.mark.skip(reason="Requires large dataset download, not suitable for CI")
def test_read_camelsh_kr_attr_xrdataset():
    ds = CamelshKr(data_path)
    result_1 = ds.read_attr_xrdataset(gage_id_lst="1001655", var_lst=["p_mean"])[
        "p_mean"
    ].values
    csv_path = os.path.join(
        data_path,
        "CAMELSH_KR",
        "attributes_climate_ERA5Land.csv",
    )
    df = pd.read_csv(csv_path)
    result_2 = df[df["STAID"] == 1001655]["p_mean"].values[0]
    assert result_1 == result_2


# "Test whether read_ts_xrdataset() from camels_sk correctly reads .nc files and returns a list of watershed ID strings."
@pytest.mark.skip(reason="Requires large dataset download, not suitable for CI")
def test_read_camelsh_kr_timeseries_xrdataset():
    ds = CamelshKr(data_path)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["1001655"],
        var_lst=["total_precipitation"],
        t_range=["2001-01-04", "2001-01-04"],
    )
    station_data = ts_data["total_precipitation"]
    result_1 = station_data.values.flatten()[0]
    file_path = os.path.join(
        data_path,
        "CAMELSH_KR",
        "timeseries",
        "timeseries",
        "1001655.csv",
    )
    ds = pd.read_csv(file_path)
    result_2 = ds[ds["DateTime"] == "04-Jan-2001 00:00:00"][
        "total_precipitation"
    ].values[0]
    assert result_1 == result_2


# "Test whether read_attr_xrdataset() from camels_gb correctly reads .nc files and returns a list of watershed ID strings."
@pytest.mark.skip(reason="Requires large dataset download, not suitable for CI")
def test_read_camels_gb_attr_xrdataset():
    ds = CamelsGb(data_path)
    result_1 = ds.read_attr_xrdataset(gage_id_lst="102001", var_lst=["p_mean"])[
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
@pytest.mark.skip(reason="Requires large dataset download, not suitable for CI")
def test_read_camels_gb_timeseries_xrdataset():
    ds = CamelsGb(data_path)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["102001"],
        var_lst=["pcp_mm"],
        t_range=["1981-01-04", "1981-01-04"],
    )
    station_data = ts_data["pcp_mm"]
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
    ds = pd.read_csv(file_path)
    result_2 = ds[ds["date"] == "1981-01-04"]["precipitation"].values[0]
    assert result_1 == result_2


# "Test whether read_attr_xrdataset() from camels_fi correctly reads .nc files and returns a list of watershed ID strings."
@pytest.mark.skip(reason="Requires large dataset download, not suitable for CI")
def test_read_camels_fi_attr_xrdataset():
    ds = CamelsFi(data_path)
    result_1 = ds.read_attr_xrdataset(gage_id_lst="1012", var_lst=["p_mean"])[
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
@pytest.mark.skip(reason="Requires large dataset download, not suitable for CI")
def test_read_camels_fi_timeseries_xrdataset():
    ds = CamelsFi(data_path)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["1012"],
        var_lst=["pcp_mm"],
        t_range=["1981-01-04", "1981-01-04"],
    )
    station_data = ts_data["pcp_mm"]
    result_1 = station_data.values.flatten()
    file_path = os.path.join(
        data_path,
        "CAMELS_FI",
        "CAMELS-FI",
        "CAMELS-FI",
        "data",
        "timeseries",
        "CAMELS_FI_hydromet_timeseries_1012_19610101-20231231.csv",
    )
    ds = pd.read_csv(file_path)
    result_2 = ds[ds["date"] == "1981-01-04"]["precipitation"].values[0]
    assert result_1 == result_2


# "Test whether read_attr_xrdataset() from camels_lux correctly reads .nc files and returns a list of watershed ID strings."
@pytest.mark.skip(reason="Requires large dataset download, not suitable for CI")
def test_read_camels_lux_attr_xrdataset():
    ds = CamelsLux(data_path)
    result_1 = ds.read_attr_xrdataset(gage_id_lst="ID_05", var_lst=["Qspec_sum"])[
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
@pytest.mark.skip(reason="Requires large dataset download, not suitable for CI")
def test_read_camels_lux_timeseries_xrdataset():
    ds = CamelsLux(data_path)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["ID_05"],
        var_lst=["q_cms_obs"],
        t_range=["2005-01-04", "2005-01-04"],
    )
    station_data = ts_data["q_cms_obs"]
    result_1 = station_data.values.flatten()
    file_path = os.path.join(
        data_path,
        "CAMELS_LUX",
        "CAMELS-LUX",
        "timeseries",
        "daily",
        "CAMELS_LUX_hydromet_timeseries_ID_05.csv",
    )
    ds = pd.read_csv(file_path)
    result_2 = ds[ds["Date"] == "2005-01-04"]["Q"].values[0]
    assert result_1 == result_2


# "Test whether read_attr_xrdataset() from camels_nz correctly reads .nc files and returns a list of watershed ID strings."
@pytest.mark.skip(reason="Requires large dataset download, not suitable for CI")
def test_read_camels_nz_attr_xrdataset():
    ds = CamelsNz(data_path)
    result_1 = ds.read_attr_xrdataset(gage_id_lst="3819", var_lst=["area_km2"])[
        "area_km2"
    ].values
    csv_path = os.path.join(
        data_path,
        "CAMELS_NZ",
        "CAMELS_NZ_Catchment_Atrributes",
        "1.CAMELS_NZ_Catchment_information.csv",
    )
    df = pd.read_csv(csv_path)

    result_2 = df[df["Station_ID"] == 3819]["uparea"].values[0]
    assert result_1 == result_2


# "Test whether read_ts_xrdataset() from camels_nz correctly reads .nc files and returns a list of watershed ID strings."
@pytest.mark.skip(reason="Requires large dataset download, not suitable for CI")
def test_read_camels_nz_timeseries_xrdataset():
    ds = CamelsNz(data_path)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["3819"],
        var_lst=["q_cms_obs"],
        t_range=["1981-01-04", "1981-01-04"],
    )
    station_data = ts_data["q_cms_obs"]
    result_1 = station_data.values.flatten()[0]
    file_path = os.path.join(
        data_path,
        "CAMELS_NZ",
        "CAMELS_NZ_Streamflow",
        "flow_station_id_3819.csv",
    )
    ds = pd.read_csv(file_path)
    result_2 = ds[ds["time"] == "1981-01-04 00:00:00"]["flow"].values[0]
    assert np.isclose(result_1, result_2)


# "Test whether read_attr_xrdataset() from camels_de correctly reads .nc files and returns a list of watershed ID strings."
@pytest.mark.skip(reason="Requires large dataset download, not suitable for CI")
def test_read_camels_de_attr_xrdataset():
    ds = CamelsDe(data_path)
    result_1 = ds.read_attr_xrdataset(gage_id_lst="DE110010", var_lst=["p_mean"])[
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
@pytest.mark.skip(reason="Requires large dataset download, not suitable for CI")
def test_read_camels_de_timeseries_xrdataset():
    ds = CamelsDe(data_path)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["DE110010"],
        var_lst=["q_cms_obs"],
        t_range=["1981-01-04", "1981-01-04"],
    )
    station_data = ts_data["q_cms_obs"]
    result_1 = station_data.values.flatten()
    file_path = os.path.join(
        data_path,
        "CAMELS_DE",
        "camels_de",
        "timeseries",
        "CAMELS_DE_hydromet_timeseries_DE110010.csv",
    )
    ds = pd.read_csv(file_path)
    result_2 = ds[ds["date"] == "1981-01-04"]["discharge_vol"].values[0]
    assert result_1 == result_2


# "Test whether read_attr_xrdataset() from camels_fr correctly reads .nc files and returns a list of watershed ID strings."
@pytest.mark.skip(reason="Requires large dataset download, not suitable for CI")
def test_read_camels_fr_attr_xrdataset():
    ds = CamelsFr(data_path)
    result_1 = ds.read_attr_xrdataset(
        gage_id_lst="A140202001", var_lst=["hgl_krs_not_karstic"]
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
@pytest.mark.skip(reason="Requires large dataset download, not suitable for CI")
def test_read_camels_fr_timeseries_xrdataset():
    ds = CamelsFr(data_path)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["A140202001"],
        var_lst=["q_cms_obs"],
        t_range=["1981-01-04", "1981-01-04"],
    )
    station_data = ts_data["q_cms_obs"]
    result_1 = station_data.values.flatten()
    file_path = os.path.join(
        data_path,
        "CAMELS_FR",
        "CAMELS_FR_time_series",
        "CAMELS_FR_time_series",
        "daily",
        "CAMELS_FR_tsd_A140202001.csv",
    )
    ds = pd.read_csv(file_path, sep=";", skiprows=7, header=0)
    result_2 = ds[ds["tsd_date"] == 19810104]["tsd_q_l"].values[0]
    assert result_1 == result_2


# "Test whether read_attr_xrdataset() from camels_ch correctly reads .nc files and returns a list of watershed ID strings."
@pytest.mark.skip(reason="Requires large dataset download, not suitable for CI")
def test_read_camels_ch_attr_xrdataset():
    ds = CamelsCh(data_path)
    result_1 = ds.read_attr_xrdataset(gage_id_lst="2109", var_lst=["p_mean"])[
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
    df = pd.read_csv(csv_path, sep=";", skiprows=1, header=0)

    result_2 = df[df["gauge_id"] == 2109]["p_mean"].values[0]
    assert result_1 == result_2


# "Test whether read_ts_xrdataset() from camels_ch correctly reads .nc files and returns a list of watershed ID strings."
@pytest.mark.skip(reason="Requires large dataset download, not suitable for CI")
def test_read_camels_ch_timeseries_xrdataset():
    ds = CamelsCh(data_path)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["2109"],
        var_lst=["pcp_mm"],
        t_range=["1981-01-04", "1981-01-04"],
    )
    station_data = ts_data["pcp_mm"]
    result_1 = station_data.values.flatten()
    file_path = os.path.join(
        data_path,
        "CAMELS_CH",
        "camels_ch",
        "camels_ch",
        "time_series",
        "observation_based",
        "CAMELS_CH_obs_based_2109.csv",
    )
    ds = pd.read_csv(file_path, sep=";", header=0)
    result_2 = ds[ds["date"] == "1981-01-04"]["precipitation(mm/d)"].values[0]
    assert result_1 == result_2
