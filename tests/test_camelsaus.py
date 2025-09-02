from hydrodataset.camels_aus_aqua import CamelsAus
import xarray as xr
from hydrodataset import CACHE_DIR, SETTING
import numpy as np
import pandas as pd
import os

data_path = SETTING["local_data_path"]["root"]


# 数据集文件位于data_path内的CAMELS_AUS文件夹
# "Test whether read_attr_xrdataset() correctly reads .nc files and returns a list of watershed ID strings."
def test_read_attr_xrdataset():
    ds = CamelsAus(data_path)
    result_1 = ds.read_attr_xrdataset(gage_id_lst='912105A', var_lst=["anngro_mega"])[
        'anngro_mega'
    ].values
    csv_path = os.path.join(
        data_path,
        "CAMELS_AUS",
        "04_attributes",
        "04_attributes",
        "CatchmentAttributes_05_Other.csv",
    )
    df = pd.read_csv(csv_path)
    # 获取STAID为01011000的p_mean值
    result_2 = df[df['station_id'] == '912105A']['anngro_mega'].values[0]
    assert result_1 == result_2


# q_cms_obs存在单位换算
# "Test whether read_ts_xrdataset() correctly reads .nc files and returns a list of watershed ID strings."
def test_read_timeseries_xrdataset():
    ds = CamelsAus(data_path)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=['912105A'],
        var_lst=['airtemp_C_silo_min'],
        t_range=['1980-01-04', '1980-01-04'],
    )
    station_data = ts_data['airtemp_C_silo_min']
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
        (ds['year'] == 1980) & (ds['month'] == 1) & (ds['day'] == 4), '912105A'
    ].values[0]
    assert result_1 == result_2
