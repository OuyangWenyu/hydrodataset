import os
import xarray as xr
import numpy as np
from tqdm import tqdm
from hydrodataset import HydroDataset, StandardVariable
from aqua_fetch import CAMELSH


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
        super().__init__(data_path)
        self.region = region
        self.download = download
        self.aqua_fetch = CAMELSH(data_path)

    @property
    def _attributes_cache_filename(self):
        return "camelsh_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camelsh_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1980-01-01", "2024-12-31"]

    _subclass_static_definitions = {
        # 基本站点信息
        "area": {"specific_name": "area_km2", "unit": "km^2"},
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
       
    }
    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "nldas",
            "sources": {
                "nldas": {"specific_name": "q_cms_obs", "unit": "m^3/s"}
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "nldas",
            "sources": {
                "nldas": {"specific_name": "pcp_mm", "unit": "mm/hour"},
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "nldas",
            "sources": {
                "nldas": {"specific_name": "airtemp_c_mean", "unit": "°C"},
            },
        },
        StandardVariable.LONGWAVE_SOLAR_RADIATION: {
            "default_source": "nldas",
            "sources": {
                "nldas": {"specific_name": "lwdown", "unit": "W/m^2"},
            },
        },
        # Shortwave radiation flux downwards (surface)
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "nldas",
            "sources": {
                "nldas": {"specific_name": "swdown", "unit": "W/m^2"},
            },
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "nldas",
            "sources": {
                "nldas": {"specific_name": "pet_mm", "unit": "mm/hour"}
            },
        },
        StandardVariable.SURFACE_PRESSURE: {
            "default_source": "nldas",
            "sources": {
                "nldas": {"specific_name": "psurf", "unit": "Pa"},
            },
        },
        # 10-meter above ground Zonal wind speed
        StandardVariable.WIND_SPEED: {
            "default_source": "nldas",
            "sources": {
                "nldas": {"specific_name": "wind_e", "unit": "m/s"},
            },
        },
        StandardVariable.MERIDIONAL_WIND_SPEED: {
            "default_source": "nldas",
            "sources": {
                "nldas": {"specific_name": "wind_n", "unit": "m/s"},
            },
        },
        StandardVariable.RELATIVE_HUMIDITY: {
            "default_source": "nldas",
            "sources": {
                "nldas": {"specific_name": "qair", "unit": "kg/kg"},
            },
        },
        StandardVariable.WATER_LEVEL: {
            "default_source": "nldas",
            "sources": {
                "nldas": {"specific_name": "water_level", "unit": "m"},
            },
        },
        StandardVariable.CAPE: {
            "default_source": "nldas",
            "sources": {
                "nldas": {"specific_name": "cape", "unit": "J/kg"},
            },
        },
        StandardVariable.CRAINF_FRAC: {
            "default_source": "nldas",
            "sources": {
                "nldas": {"specific_name": "crainf_frac", "unit": "Fraction"},
            },
        },
    }

    def cache_timeseries_xrdataset(self, batch_size=100):
        """
        分批缓存时间序列数据到NetCDF文件，每批保存为独立文件

        Args:
            batch_size: 每批处理的站点数量，默认100个站点
        """
        if not hasattr(self, "aqua_fetch"):
            raise NotImplementedError("aqua_fetch attribute is required")

        # 构建变量名到单位的映射
        unit_lookup = {}
        if hasattr(self, "_dynamic_variable_mapping"):
            for std_name, mapping_info in self._dynamic_variable_mapping.items():
                for source, source_info in mapping_info["sources"].items():
                    unit_lookup[source_info["specific_name"]] = source_info["unit"]

        # 获取所有站点ID
        gage_id_lst = self.read_object_ids().tolist()
        total_stations = len(gage_id_lst)

        # 获取原始变量列表并清理
        original_var_lst = self.aqua_fetch.dynamic_features
        cleaned_var_lst = self._clean_feature_names(original_var_lst)
        var_name_mapping = dict(zip(original_var_lst, cleaned_var_lst))

        print(f"开始分批处理 {total_stations} 个站点，每批 {batch_size} 个站点")
        print(f"总批次数: {(total_stations + batch_size - 1)//batch_size}")

        # 确保缓存目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 分批处理站点并独立保存
        batch_num = 1
        for batch_idx in range(0, total_stations, batch_size):
            batch_end = min(batch_idx + batch_size, total_stations)
            batch_stations = gage_id_lst[batch_idx:batch_end]

            print(f"\n处理批次 {batch_num}/{(total_stations + batch_size - 1)//batch_size}")
            print(f"站点范围: {batch_idx} - {batch_end-1} (共 {len(batch_stations)} 个站点)")

            try:
                # 获取本批次数据
                batch_data = self.aqua_fetch.fetch_stations_features(
                    stations=batch_stations,
                    dynamic_features=original_var_lst,
                    static_features=None,
                    st=self.default_t_range[0],
                    en=self.default_t_range[1],
                    as_dataframe=False,
                )

                dynamic_data = (
                    batch_data[1] if isinstance(batch_data, tuple) else batch_data
                )

                # 处理变量
                new_data_vars = {}
                time_coord = dynamic_data.coords["time"]

                for original_var in tqdm(
                    original_var_lst,
                    desc=f"处理变量 (批次 {batch_num})",
                    total=len(original_var_lst),
                ):
                    cleaned_var = var_name_mapping[original_var]
                    var_data = []
                    for station in batch_stations:
                        if station in dynamic_data.data_vars:
                            station_data = dynamic_data[station].sel(
                                dynamic_features=original_var
                            )
                            if "dynamic_features" in station_data.coords:
                                station_data = station_data.drop("dynamic_features")
                            var_data.append(station_data)

                    if var_data:
                        combined = xr.concat(var_data, dim="basin")
                        combined["basin"] = batch_stations
                        combined.attrs["units"] = unit_lookup.get(cleaned_var, "unknown")
                        new_data_vars[cleaned_var] = combined

                # 创建本批次的Dataset
                batch_ds = xr.Dataset(
                    data_vars=new_data_vars,
                    coords={
                        "basin": batch_stations,
                        "time": time_coord,
                    },
                )

                # 保存本批次到独立文件
                batch_filename = f"batch{batch_num:03d}_camelsh_timeseries.nc"
                batch_filepath = self.cache_dir.joinpath(batch_filename)

                print(f"保存批次 {batch_num} 到: {batch_filepath}")
                batch_ds.to_netcdf(batch_filepath)
                print(f"批次 {batch_num} 保存成功")

            except Exception as e:
                print(f"批次 {batch_num} 处理失败: {e}")
                import traceback
                traceback.print_exc()
                continue

            batch_num += 1

        print(f"\n所有批次处理完成！共保存 {batch_num - 1} 个批次文件")

    def read_ts_xrdataset(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        sources: dict = None,
        **kwargs,
    ):
        """
        读取时间序列数据（支持标准化变量名和多数据源）

        从分批保存的缓存文件中读取数据

        Args:
            gage_id_lst: 站点ID列表
            t_range: 时间范围 [start, end]
            var_lst: 标准变量名列表
            sources: 数据源字典，格式为 {变量名: 数据源} 或 {变量名: [数据源列表]}

        Returns:
            xr.Dataset: 包含请求数据的xarray数据集
        """
        if (
            not hasattr(self, "_dynamic_variable_mapping")
            or not self._dynamic_variable_mapping
        ):
            raise NotImplementedError(
                "This dataset does not support the standardized variable mapping."
            )

        if var_lst is None:
            var_lst = list(self._dynamic_variable_mapping.keys())

        if t_range is None:
            t_range = self.default_t_range

        target_vars_to_fetch = []
        rename_map = {}

        # 处理变量名映射和数据源选择
        for std_name in var_lst:
            if std_name not in self._dynamic_variable_mapping:
                raise ValueError(
                    f"'{std_name}' is not a recognized standard variable for this dataset."
                )

            mapping_info = self._dynamic_variable_mapping[std_name]

            # 确定使用哪个或哪些数据源
            is_explicit_source = sources and std_name in sources
            sources_to_use = []
            if is_explicit_source:
                provided_sources = sources[std_name]
                if isinstance(provided_sources, list):
                    sources_to_use.extend(provided_sources)
                else:
                    sources_to_use.append(provided_sources)
            else:
                sources_to_use.append(mapping_info["default_source"])

            # 只有在用户显式请求多个数据源时才需要后缀
            needs_suffix = is_explicit_source and len(sources_to_use) > 1
            for source in sources_to_use:
                if source not in mapping_info["sources"]:
                    raise ValueError(
                        f"Source '{source}' is not available for variable '{std_name}'."
                    )

                actual_var_name = mapping_info["sources"][source]["specific_name"]
                target_vars_to_fetch.append(actual_var_name)
                output_name = f"{std_name}_{source}" if needs_suffix else std_name
                rename_map[actual_var_name] = output_name

        # 查找所有批次文件
        import glob
        batch_pattern = str(self.cache_dir / "batch*_camelsh_timeseries.nc")
        batch_files = sorted(glob.glob(batch_pattern))

        if not batch_files:
            print("未找到批次缓存文件，开始创建缓存...")
            self.cache_timeseries_xrdataset()
            batch_files = sorted(glob.glob(batch_pattern))

            if not batch_files:
                raise FileNotFoundError("缓存创建失败，未找到批次文件")

        print(f"找到 {len(batch_files)} 个批次文件")

        # 如果没有指定站点，则需要读取所有批次
        if gage_id_lst is None:
            print("未指定站点列表，将读取所有站点...")
            gage_id_lst = self.read_object_ids().tolist()

        # 将站点ID转为字符串（保证一致性）
        gage_id_lst = [str(gid) for gid in gage_id_lst]

        # 遍历批次文件，找到包含所需站点的批次
        relevant_datasets = []
        for batch_file in batch_files:
            try:
                # 先只打开坐标，不加载数据
                ds_batch = xr.open_dataset(batch_file)
                batch_basins = [str(b) for b in ds_batch.basin.values]

                # 检查此批次是否包含所需站点
                common_basins = list(set(gage_id_lst) & set(batch_basins))

                if common_basins:
                    print(f"批次 {os.path.basename(batch_file)}: 包含 {len(common_basins)} 个所需站点")

                    # 检查变量是否存在
                    missing_vars = [v for v in target_vars_to_fetch if v not in ds_batch.data_vars]
                    if missing_vars:
                        ds_batch.close()
                        raise ValueError(
                            f"批次 {os.path.basename(batch_file)} 缺少变量: {missing_vars}"
                        )

                    # 选择变量和站点
                    ds_subset = ds_batch[target_vars_to_fetch]
                    ds_selected = ds_subset.sel(
                        basin=common_basins,
                        time=slice(t_range[0], t_range[1])
                    )

                    relevant_datasets.append(ds_selected)
                    ds_batch.close()
                else:
                    ds_batch.close()

            except Exception as e:
                print(f"读取批次文件 {batch_file} 失败: {e}")
                continue

        if not relevant_datasets:
            raise ValueError(f"在所有批次文件中未找到指定站点: {gage_id_lst}")

        print(f"从 {len(relevant_datasets)} 个批次中读取数据...")

        # 合并所有相关批次的数据
        if len(relevant_datasets) == 1:
            final_ds = relevant_datasets[0]
        else:
            final_ds = xr.concat(relevant_datasets, dim="basin")

        # 重命名为标准变量名
        final_ds = final_ds.rename(rename_map)

        # 确保按照输入顺序排列站点
        if len(gage_id_lst) > 0:
            # 只选择实际存在的站点
            existing_basins = [b for b in gage_id_lst if b in final_ds.basin.values]
            if existing_basins:
                final_ds = final_ds.sel(basin=existing_basins)

        return final_ds

































'''
    def _get_attribute_units(self):
        return {
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

    def _get_timeseries_units(self):
        return [
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
            "W/m²​​ ",
        ]
'''
