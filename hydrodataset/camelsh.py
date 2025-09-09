import os
import glob
import zipfile
import pandas as pd
import xarray as xr
from collections import OrderedDict
from hydrodataset import HydroDataset
from tqdm import tqdm
import numpy as np
from typing import Union
from hydroutils import hydro_time, hydro_file
from hydrodataset import CACHE_DIR, HydroDataset, CAMELS_REGIONS
import json
import collections
from pathlib import Path
from aqua_fetch import CAMELSH
from hydrodataset import CACHE_DIR, SETTING
import warnings
import re


class Camelsh(HydroDataset):
    """CAMELSH (CAMELS-Hourly) dataset class extending RainfallRunoff.

    This class provides access to the CAMELSH dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    def __init__(self, data_path, region=None, download=False):
        """Initialize CAMELSH dataset.

        Args:
            data_path: Path to the CAMELSH data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
        """
        # Call parent class RainfallRunoff constructor with CAMELSH dataset
        # Set additional attributes
        self.data_path = data_path
        self.region = region
        self.download = download
        self.aqua_fetch = CAMELSH(data_path)
        self.set_data_source_describe()

    def set_data_source_describe(self):
        """Set up dataset file path descriptions.

        Configures paths for various dataset components including timeseries,
        attributes, shapefiles, site info, and hourly data files.
        """
        self.ds_description = OrderedDict()
        self.ds_description["timeseries_dir"] = os.path.join(
            self.data_path,
            "CAMELSH",
            "timeseries",
            "Data",
            "CAMELSH",
            "timeseries",
        )  # timeseries_nonobs
        self.ds_description["attributes_file"] = os.path.join(
            self.data_path, "CAMELSH", "attributes"
        )
        self.ds_description["shapefile_dir"] = os.path.join(
            self.data_path, "CAMELSH", "shapefiles"
        )
        self.ds_description["site_info_file"] = os.path.join(
            self.data_path, "CAMELSH", "info.csv"
        )
        self.ds_description["Hourly2_file"] = os.path.join(
            self.data_path, "CAMELSH", "Hourly2", "Hourly2"
        )

    def read_object_ids(self) -> np.ndarray:
        """Read watershed station ID list.

        Uses parent class RainfallRunoff's stations method to get all available station IDs.

        Returns:
            np.ndarray: Array containing all station IDs
        """
        # Call parent class stations method to get all station IDs
        stations_list = self.aqua_fetch.stations()

        # Convert to numpy array and return
        return np.array(stations_list)

    def read_ts_all(self):

        return self.aqua_fetch.dynamic_features

    def read_attr_all(self):
        return self.aqua_fetch.static_features

    def cache_attributes_xrdataset(self):
        ds_attr = self.aqua_fetch.fetch_static_features().to_xarray()
        BASE_UNITS = {
            # 地形特征
            "dis_m3_": "m^3/s",
            "run_mm_": "millimeter",
            "inu_pc_": "percent",
            "lka_pc_": "1e-1 * percent",
            "lkv_mc_": "1e6 * m^3",
            "rev_mc_": "1e6 * m^3",
            "dor_pc_": "percent (x10)",
            "ria_ha_": "hectares",
            "riv_tc_": "1e3 * m^3",
            "gwt_cm_": "centimeter",
            "ele_mt_": "meter",
            "slp_dg_": "1e-1 * degree",
            "sgr_dk_": "decimeter/km",
            "clz_cl_": "dimensionless",
            "cls_cl_": "dimensionless",
            "tmp_dc_": "degree_Celsius",
            "pre_mm_": "millimeters",
            "pet_mm_": "millimeters",
            "aet_mm_": "millimeters",
            "ari_ix_": "1e-2",
            "cmi_ix_": "1e-2",
            "snw_pc_": "percent",
            "glc_cl_": "dimensionless",
            "glc_pc_": "percent",
            "pnv_cl_": "dimensionless",
            "pnv_pc_": "percent",
            "wet_cl_": "dimensionless",
            "wet_pc_": "percent",
            "for_pc_": "percent",
            "crp_pc_": "percent",
            "pst_pc_": "percent",
            "ire_pc_": "percent",
            "gla_pc_": "percent",
            "prm_pc_": "percent",
            "pac_pc_": "percent",
            "tbi_cl_": "dimensionless",
            "tec_cl_": "dimensionless",
            "fmh_cl_": "dimensionless",
            "fec_cl_": "dimensionless",
            "cly_pc_": "percent",
            "slt_pc_": "percent",
            "snd_pc_": "percent",
            "soc_th_": "tonne/hectare",
            "swc_pc_": "percent",
            "lit_cl_": "dimensionless",
            "kar_pc_": "percent",
            "ero_kh_": "kg/hectare/year",
            "pop_ct_": "1e3",
            "ppd_pk_": "1/km^2",
            "urb_pc_": "percent",
            "nli_ix_": "1e-2",
            "rdd_mk_": "meter/km^2",
            "hft_ix_": "1e-1",
            "gad_id_": "dimensionless",
            "gdp_ud_": "dimensionless",
            "hdi_ix_": "1e-3",
        }

        def get_unit_by_prefix(var_name):
            """通过前缀匹配基础单位"""
            for prefix, unit in BASE_UNITS.items():
                if var_name.startswith(prefix):
                    return unit
            return None  # 未匹配时返回None

        # 智能单位分配函数
        def get_unit(var_name):
            """综合单位分配函数"""
            # 1. 先尝试前缀匹配
            prefix_unit = get_unit_by_prefix(var_name)
            if prefix_unit:
                return prefix_unit

            # 3. 其他匹配规则...

            return 'undefined'  # 默认值

        for var in ds_attr.data_vars:
            unit = get_unit(var)
            ds_attr[var].attrs['units'] = unit

            # 为分类变量添加描述
            if unit == 'class':
                ds_attr[var].attrs['description'] = 'Classification code'

        print("savepath:", CACHE_DIR)
        ds_attr.to_netcdf(CACHE_DIR.joinpath("camelsh_attributes.nc"))
        return

    def cache_timeseries_xrdataset(self):
        """缓存所有时间序列数据到netCDF文件，使用分批保存方式。

        此方法将所有站点的时间序列数据分批保存为netCDF文件，每个批次包含100个站点，
        数据结构为(basin, time)，其中basin是站点ID，time是时间序列。
        """
        batch_size = 100
        output_dir = CACHE_DIR

        print("开始CAMELSH时间序列数据缓存...")

        try:
            # 获取所有可用站点
            gage_id_lst = self.aqua_fetch.stations()

            # 获取动态特征列表
            var_lst = self.aqua_fetch.dynamic_features

            # 获取时间范围
            timeseries_dir = Path(self.ds_description["timeseries_dir"])
            data_files = list(timeseries_dir.glob("*.nc"))
            with xr.open_dataset(data_files[0]) as ds:
                st = str(ds.DateTime.min().values)
                en = str(ds.DateTime.max().values)

            n_stations = len(gage_id_lst)
            n_batches = (n_stations + batch_size - 1) // batch_size

            print(f"处理 {n_stations} 个站点，分为 {n_batches} 批...")

            # 定义单位列表（按变量顺序）
            units = [
                "mm/day",
                "m",
                "°C",
                "kg/kg",
                "Pa",
                "m/s",
                "m/s",
                "W/m^2",
                "​​Fraction",
                "​​J/kg​​ ",
                "kg/m^2",
                "kg/m^2",
                'W/m²​​ ',
            ]

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_stations)
                batch_stations = gage_id_lst[start_idx:end_idx]

                print(
                    f"处理批次 {batch_idx + 1}/{n_batches} (stations {start_idx + 1}-{end_idx})"
                )

                # 获取批次数据
                batch_data = self.aqua_fetch.fetch_stations_features(
                    stations=batch_stations,
                    dynamic_features=var_lst,
                    static_features=None,
                    st=st,
                    en=en,
                    as_dataframe=False,
                )

                # 提取动态数据
                dynamic_data = (
                    batch_data[1] if isinstance(batch_data, tuple) else batch_data
                )

                # 转换为目标结构
                new_data_vars = {}
                # 获取时间坐标（从原始数据中提取）
                time_coord = dynamic_data.coords['time']

                for var_idx, var_name in enumerate(var_lst):
                    var_data = []
                    for station in batch_stations:
                        if station in dynamic_data.data_vars:
                            # 提取变量数据并移除dynamic_features坐标
                            station_data = dynamic_data[station].sel(
                                dynamic_features=var_name
                            )
                            if 'dynamic_features' in station_data.coords:
                                station_data = station_data.drop('dynamic_features')
                            var_data.append(station_data)

                    if var_data:
                        combined = xr.concat(var_data, dim='basin')
                        combined['basin'] = batch_stations
                        combined.attrs['units'] = (
                            units[var_idx] if var_idx < len(units) else 'unknown'
                        )
                        new_data_vars[var_name] = combined

                # 创建新数据集
                new_ds = xr.Dataset(
                    data_vars=new_data_vars,
                    coords={
                        'basin': batch_stations,
                        'time': time_coord,
                    },
                )

                # 添加全局属性
                new_ds.attrs.update(
                    {
                        'title': 'CAMELSH Dataset',
                        'batch': f'{batch_idx + 1}/{n_batches}',
                        'stations': f'{start_idx + 1}-{end_idx}',
                    }
                )

                # 保存文件
                batch_filename = f"camelsh_timeseries_batch_{batch_idx + 1:03d}.nc"
                batch_filepath = CACHE_DIR / batch_filename

                # 确保缓存目录存在且有写入权限

                batch_filepath.parent.mkdir(parents=True, exist_ok=True)
                new_ds.to_netcdf(batch_filepath)
                print(f"成功保存批次 {batch_idx + 1} 到: {batch_filepath}")

            print("所有批次处理完成！")

        except Exception as e:
            print(f"缓存失败: {e}")
            raise

        return {"status": "success", "batches": n_batches}

    def read_attr_xrdataset(self, gage_id_lst=None, var_lst=None, **kwargs):

        try:
            attr = xr.open_dataset(CACHE_DIR.joinpath("camelsh_attributes.nc"))
        except FileNotFoundError:
            self.cache_attributes_xrdataset()
            attr = xr.open_dataset(CACHE_DIR.joinpath("camelsh_attributes.nc"))
        if var_lst is None or len(var_lst) == 0:
            var_lst = self.read_attr_all()
            return attr[var_lst].sel(STAID=gage_id_lst)
        else:
            return attr[var_lst].sel(STAID=gage_id_lst)

    def read_ts_xrdataset(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs,
    ):
        """从分批保存的netCDF文件中读取时间序列数据。

        根据gage_id_lst遍历所有批次文件，找到包含目标站点的数据并合并。

        Args:
            gage_id_lst: 站点ID列表，如果为None则返回所有站点
            t_range: 时间范围 [start_time, end_time]，如果为None则返回所有时间
            var_lst: 变量列表，如果为None则返回所有变量
            **kwargs: 其他参数

        Returns:
            xarray.Dataset: 包含指定站点、变量和时间范围的数据
        """
        if var_lst is None:
            var_lst = self.read_ts_all()
        if t_range is None:
            t_range = ["1980-01-01", "2024-12-31"]
        # 检查缓存文件是否存在
        ts_files = sorted(CACHE_DIR.glob("camelsh_timeseries_batch_*.nc"))

        # 如果没有缓存文件，自动创建
        if not ts_files:
            print("未找到缓存文件，正在创建时间序列缓存...")
            self.cache_timeseries_xrdataset()
            ts_files = sorted(CACHE_DIR.glob("camelsh_timeseries_batch_*.nc"))
            if not ts_files:
                raise FileNotFoundError("创建缓存文件失败")

        # 初始化结果数据集
        result_ds = xr.Dataset()

        for batch_file in ts_files:
            with xr.open_dataset(batch_file) as batch_ds:
                # 站点筛选
                if gage_id_lst is not None and len(gage_id_lst) > 0:
                    batch_ds = batch_ds.sel(
                        basin=[gid for gid in gage_id_lst if gid in batch_ds.basin]
                    )

                # 变量筛选
                if var_lst:
                    batch_ds = batch_ds[[v for v in var_lst if v in batch_ds.data_vars]]

                # 时间筛选
                if t_range:
                    batch_ds = batch_ds.sel(time=slice(*t_range))

                result_ds = xr.merge([result_ds, batch_ds])

        # 验证结果
        if not result_ds.data_vars:
            raise ValueError("未找到匹配的数据，请检查参数")

        return result_ds
