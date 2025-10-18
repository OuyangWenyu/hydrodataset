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
from water_datasets import CAMELS_NZ
from hydrodataset import CACHE_DIR, SETTING
import warnings
import re


class CamelsNz(HydroDataset):
    """CAMELS_NZ dataset class extending RainfallRunoff.

    This class provides access to the CAMELS_NZ dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    def __init__(self, data_path, region=None, download=False):
        """Initialize CAMELS_NZ dataset.

        Args:
            data_path: Path to the CAMELS_NZ data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
        """
        # Call parent class RainfallRunoff constructor with CAMELS_NZ dataset
        # Set additional attributes
        self.data_path = data_path
        self.region = region
        self.download = download
        self.aqua_fetch = CAMELS_NZ(data_path)

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
        ds_attr.to_netcdf(CACHE_DIR.joinpath("camels_nz_attributes.nc"))
        return

    def cache_timeseries_xrdataset(self):

        output_dir = CACHE_DIR

        gage_id_lst = self.aqua_fetch.stations()

        # 获取动态特征列表
        var_lst = self.aqua_fetch.dynamic_features

        units = [
            "mm/day",  # pet_mm
            "mm/day",  # pcp_mm
            "%",  # rh_%
            "°C",  # airtemp_C_mean K->°C
            "m^3/s",  # q_cms_obs
        ]

        batch_data = self.aqua_fetch.fetch_stations_features(
            stations=gage_id_lst,
            dynamic_features=var_lst,
            static_features=None,
            st='1972-01-01 00:00:00',
            en='2024-08-02 09:00:00',
            as_dataframe=False,
        )

        dynamic_data = batch_data[1] if isinstance(batch_data, tuple) else batch_data

        # 转换为目标结构
        new_data_vars = {}
        # 获取时间坐标（从原始数据中提取）
        time_coord = dynamic_data.coords['time']

        for var_idx, var_name in enumerate(var_lst):
            var_data = []
            for station in gage_id_lst:
                if station in dynamic_data.data_vars:
                    # 提取变量数据并移除dynamic_features坐标
                    station_data = dynamic_data[station].sel(dynamic_features=var_name)
                    if 'dynamic_features' in station_data.coords:
                        station_data = station_data.drop('dynamic_features')
                    var_data.append(station_data)

            if var_data:
                combined = xr.concat(var_data, dim='basin')
                combined['basin'] = gage_id_lst
                combined.attrs['units'] = (
                    units[var_idx] if var_idx < len(units) else 'unknown'
                )
                new_data_vars[var_name] = combined

                # 创建新数据集
        new_ds = xr.Dataset(
            data_vars=new_data_vars,
            coords={
                'basin': gage_id_lst,
                'time': time_coord,
            },
        )

        # 保存文件
        batch_filename = f"camels_nz_timeseries.nc"
        batch_filepath = CACHE_DIR / batch_filename

        # 确保缓存目录存在且有写入权限

        batch_filepath.parent.mkdir(parents=True, exist_ok=True)
        new_ds.to_netcdf(batch_filepath)
        print(f"成功保存到: {batch_filepath}")

    def read_attr_xrdataset(self, gage_id_lst=None, var_lst=None, **kwargs):

        try:
            attr = xr.open_dataset(CACHE_DIR.joinpath("camels_nz_attributes.nc"))
        except FileNotFoundError:
            self.cache_attributes_xrdataset()
            attr = xr.open_dataset(CACHE_DIR.joinpath("camels_nz_attributes.nc"))
        if var_lst is None or len(var_lst) == 0:
            var_lst = self.read_attr_all()
            return attr[var_lst].sel(Station_ID=gage_id_lst)
        else:
            return attr[var_lst].sel(Station_ID=gage_id_lst)

    def read_ts_xrdataset(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs,
    ):
        if var_lst is None:
            var_lst = self.read_ts_all()
        if t_range is None:
            t_range = ["1972-01-01 00:00:00", "2024-08-02 09:00:00"]
        camels_nz_tsnc = CACHE_DIR.joinpath("camels_nz_timeseries.nc")
        if not os.path.isfile(camels_nz_tsnc):
            self.cache_timeseries_xrdataset()
        ts = xr.open_dataset(camels_nz_tsnc)
        all_vars = ts.data_vars
        if any(var not in ts.variables for var in var_lst):
            raise ValueError(f"var_lst must all be in {all_vars}")
        return ts[var_lst].sel(basin=gage_id_lst, time=slice(t_range[0], t_range[1]))
