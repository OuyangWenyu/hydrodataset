import netCDF4 as nc
import sys
import numpy as np
import xarray as xr

# 设置输出编码为UTF-8

file_path1 = r"D:\data\stations (2)\songliaorrevent_basin_station_mapping.nc"
file_path2 = r"D:\data\stations (2)\songliaorrevent_stations_3h_batch_11047818_11049588.nc"
file_path3 = r"D:\data\stations (2)\songliaorrevent_stations_3h_batch_11101600_11120011.nc"

print(f"正在打开文件")
dataset1 = xr.open_dataset(file_path1)

dataset2 = xr.open_dataset(file_path2)

dataset3 = xr.open_dataset(file_path3)
        # 打印完整结构
print(dataset1)
print("======================================")
print(dataset2)
print("======================================")
print(dataset3)
# 查看前10个样本
print("=" * 60)
print("数据集前10个样本值")
print("=" * 60)

# 方法1: 直接查看前10行
sample_data = dataset1.isel(mapping_id=slice(0, 20))
print(sample_data)

# 方法2: 转换为DataFrame查看（更直观）
import pandas as pd
df_sample = sample_data.to_dataframe()
print("\n📊 表格形式显示:")
print(df_sample)

# 查看前5个时间步的所有站点数据
print("=" * 60)
print("前5个时间步的样本值")
print("=" * 60)

sample_data2 = dataset2.isel(time=slice(0, 5))
df_sample2 = sample_data2.to_dataframe()
print("\n📊 表格形式显示:")
print(df_sample2)