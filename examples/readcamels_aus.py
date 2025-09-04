from hydrodataset.camels_aus_aqua import CamelsAus
import pandas as pd
import os
from hydrodataset import CACHE_DIR, SETTING
import xarray as xr
import netCDF4 as nc
import numpy as np

data_path = SETTING["local_data_path"]["root"]  # ← 修改为你的实际路径


def main():
    ds = CamelsAus(data_path)
    gage_ids = ds.read_object_ids()
    print(gage_ids)
    ts_all = ds.read_ts_all()
    print(ts_all)
    '''
    ds.cache_attributes_xrdataset()
    dx = xr.open_dataset(CACHE_DIR.joinpath("camels_aus_attributes.nc"))
    print(dx)
    for var in dx.variables:
        print(f"变量名: {var}")
        if hasattr(dx[var], 'units'):
            print(f"单位: {dx[var].units}")
        else:
            print("无单位属性")
    '''
    '''
    ds.cache_timeseries_xrdataset()
    dx = xr.open_dataset(CACHE_DIR.joinpath("camels_aus_timeseries.nc"))
    print(dx)
    for var in dx.variables:
        print(f"变量名: {var}")
        if hasattr(dx[var], 'units'):
            print(f"单位: {dx[var].units}")
        else:
            print("无单位属性")
    '''

    # dx = xr.open_dataset(CACHE_DIR.joinpath("camels_aus_attributes.nc"))
    dx = ds.read_ts_xrdataset(
        gage_id_lst=gage_ids[:2],
        t_range=["1980-01-01", "1980-01-01"],
    )
    print(dx)
    '''
    dy = ds.read_ts_xrdataset(
        gage_id_lst=gage_ids[:2],
        t_range=["1990-01-01", "2000-01-01"],
        var_lst=["q_cms_obs"],
    )
    print(dy)
    '''
    '''
    dx = xr.open_dataset(CACHE_DIR.joinpath("camels_aus_timeseries.nc"))
    value = dx['q_cms_obs'].sel(time='1980-01-08', basin='912105A').values.item()
    print(value)
    print('--------------------------------')
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=['912105A'],
        var_lst=[
            'q_cms_obs',
        ],
        t_range=['1980-01-08', '1980-01-08'],
    )
    station_data = ts_data['q_cms_obs']
    result_1 = station_data.values.flatten()
    print(result_1)
    print('--------------------------------')
    file_path = r"D:/data/CAMELS_AUS/03_streamflow/03_streamflow/streamflow_MLd.csv"
    dm = pd.read_csv(file_path)
    result_2 = dm.loc[
        (dm['year'] == 1980) & (dm['month'] == 1) & (dm['day'] == 8), '912105A'
    ].values[0]
    print(result_2)
    '''


if __name__ == "__main__":
    main()
