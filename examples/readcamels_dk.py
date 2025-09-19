from hydrodataset.camels_dk_aqua import CamelsDK
import pandas as pd
import os
from hydrodataset import CACHE_DIR, SETTING
import xarray as xr
import netCDF4 as nc
import numpy as np

data_path = SETTING["local_data_path"]["root"]  # ← 修改为你的实际路径


def main():
    ds = CamelsDK(data_path)
    gage_ids = ds.read_object_ids()
    print(gage_ids)
    print('--------------------------------')
    ts_all = ds.read_ts_all()
    print(ts_all)
    print('--------------------------------')
    attr_all = ds.read_attr_all()
    print(attr_all)
    print('--------------------------------')
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=gage_ids[:2],
        t_range=["1981-01-01", "1981-01-01"],
    )
    print(ts_data)
    print('--------------------------------')
    attr_data = ds.read_attr_xrdataset(
        gage_id_lst=gage_ids[:2],
        var_lst=["p_mean"],
    )
    print(attr_data)


if __name__ == "__main__":
    main()
