from hydrodataset.camels_de_aqua import CamelsDe
import xarray as xr
from hydrodataset import CACHE_DIR, SETTING
import numpy as np
import pandas as pd
import os

data_path = SETTING["local_data_path"]["root"]


# 数据集文件位于data_path内的CAMELS_AUS文件夹
# "Test whether read_attr_xrdataset() correctly reads .nc files and returns a list of watershed ID strings."
def test_read_attr_xrdataset():
    ds = CamelsDe(data_path)
    result_1 = ds.read_attr_xrdataset(gage_id_lst='DE110010', var_lst=["p_mean"])[
        'p_mean'
    ].values
    csv_path = os.path.join(
        data_path,
        "CAMELS_DE",
        "camels_de",
        "CAMELS_DE_climatic_attributes.csv",
    )
    df = pd.read_csv(csv_path)
    result_2 = df[df['gauge_id'] == 'DE110010']['p_mean'].values[0]
    assert result_1 == result_2


# q_cms_obs存在单位换算
# "Test whether read_ts_xrdataset() correctly reads .nc files and returns a list of watershed ID strings."
def test_read_timeseries_xrdataset():
    ds = CamelsDe(data_path)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=['DE110010'],
        var_lst=['q_cms_obs'],
        t_range=['1981-01-04', '1981-01-04'],
    )
    station_data = ts_data['q_cms_obs']
    result_1 = station_data.values.flatten()
    file_path = os.path.join(
        data_path,
        "CAMELS_DE",
        "camels_de",
        "timeseries",
        "CAMELS_DE_hydromet_timeseries_DE110010.csv",
    )
    ds = pd.read_csv(file_path)
    result_2 = ds[ds['date'] == '1981-01-04']['discharge_vol'].values[0]
    assert result_1 == result_2
