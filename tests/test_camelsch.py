from hydrodataset.camels_ch_aqua import CamelsCh
import xarray as xr
from hydrodataset import CACHE_DIR, SETTING
import numpy as np
import pandas as pd
import os

data_path = SETTING["local_data_path"]["root"]


# 数据集文件位于data_path内的CAMELS_AUS文件夹
# "Test whether read_attr_xrdataset() correctly reads .nc files and returns a list of watershed ID strings."
def test_read_attr_xrdataset():
    ds = CamelsCh(data_path)
    result_1 = ds.read_attr_xrdataset(gage_id_lst='2109', var_lst=["p_mean"])[
        'p_mean'
    ].values
    csv_path = os.path.join(
        data_path,
        "CAMELS_CH",
        "camels_ch",
        "camels_ch",
        "static_attributes",
        "CAMELS_CH_climate_attributes_obs.csv",
    )
    df = pd.read_csv(
        csv_path, sep=';', skiprows=1, header=0
    )  # skiprows=1 跳过注释行，header=0 使用下一行作为列名

    # 获取 gauge_id 为 '2109' 的行的 p_mean 值（第五列）
    result_2 = df[df['gauge_id'] == 2109]['p_mean'].values[0]
    assert result_1 == result_2


# q_cms_obs存在单位换算
# "Test whether read_ts_xrdataset() correctly reads .nc files and returns a list of watershed ID strings."
def test_read_timeseries_xrdataset():
    ds = CamelsCh(data_path)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=['2109'],
        var_lst=['pcp_mm'],
        t_range=['1981-01-04', '1981-01-04'],
    )
    station_data = ts_data['pcp_mm']
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
    ds = pd.read_csv(file_path, sep=';', header=0)
    result_2 = ds[ds['date'] == '1981-01-04']['precipitation(mm/d)'].values[0]
    assert result_1 == result_2
