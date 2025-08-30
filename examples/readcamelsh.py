from hydrodataset.camelsh import Camelsh
import pandas as pd
import os
from hydrodataset import CACHE_DIR, SETTING
import xarray as xr
import netCDF4 as nc
import numpy as np

data_path = SETTING["local_data_path"]["root"]  # ← 修改为你的实际路径


def main():
    # 初始化 Camelsh 数据集
    ds = Camelsh(data_path)

    print("属性数据：")
    # ds.cache_attributes_xrdataset()
    print("流数据：")
    # ds.cache_timeseries_xrdataset()
    # 1. 获取所有站点ID

    gage_ids = ds.read_object_ids()
    result = ds.read_attr_xrdataset(gage_id_lst=gage_ids[:2], var_lst=["RIP100_81"])
    print(result['RIP100_81'].values)


if __name__ == "__main__":
    main()
