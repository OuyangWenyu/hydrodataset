"""
Author: Wenyu Ouyang
Date: 2022-09-05 23:20:24
LastEditTime: 2025-10-19 20:51:58
LastEditors: Wenyu Ouyang
Description: main modules for hydrodataset
FilePath: \hydrodataset\hydrodataset\hydro_dataset.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm
import xarray as xr

import numpy as np

from hydrodataset import ROOT_DIR, CACHE_DIR


class HydroDataset(ABC):
    """An interface for Hydrological Dataset

    For unit, we use Pint package's unit system -- unit registry

    Parameters
    ----------
    ABC : _type_
        _description_
    """

    def __init__(self, data_path, cache_path=None):
        self.data_source_dir = Path(ROOT_DIR, data_path)
        if not self.data_source_dir.is_dir():
            self.data_source_dir.mkdir(parents=True)
        if cache_path is None:
            self.cache_dir = Path(CACHE_DIR)
        else:
            self.cache_dir = Path(cache_path)
        if not self.cache_dir.is_dir():
            self.cache_dir.mkdir(parents=True)

    def get_name(self):
        raise NotImplementedError

    def set_data_source_describe(self):
        raise NotImplementedError

    def download_data_source(self):
        raise NotImplementedError

    def is_data_ready(self):
        raise NotImplementedError

    def read_object_ids(self) -> np.ndarray:
        """Read watershed station ID list."""
        if hasattr(self, "aqua_fetch"):
            stations_list = self.aqua_fetch.stations()
            return np.sort(np.array(stations_list))
        raise NotImplementedError

    def read_target_cols(
        self, gage_id_lst=None, t_range=None, target_cols=None, **kwargs
    ) -> np.ndarray:
        raise NotImplementedError

    def read_relevant_cols(
        self, gage_id_lst=None, t_range=None, var_lst=None, forcing_type=None, **kwargs
    ) -> np.ndarray:
        """3d data (site_num * time_length * var_num), time-series data"""
        raise NotImplementedError

    def read_constant_cols(
        self, gage_id_lst=None, var_lst=None, **kwargs
    ) -> np.ndarray:
        """2d data (site_num * var_num), non-time-series data"""
        raise NotImplementedError

    def read_other_cols(
        self, object_ids=None, other_cols: dict = None, **kwargs
    ) -> dict:
        """some data which cannot be easily treated as constant vars or time-series with same length as relevant vars
        CONVENTION: other_cols is a dict, where each item is also a dict with all params in it
        """
        raise NotImplementedError

    def get_constant_cols(self) -> np.ndarray:
        """the constant cols in this data_source"""
        raise NotImplementedError

    def get_relevant_cols(self) -> np.ndarray:
        """the relevant cols in this data_source"""
        raise NotImplementedError

    def get_target_cols(self) -> np.ndarray:
        """the target cols in this data_source"""
        raise NotImplementedError

    def get_other_cols(self) -> dict:
        """the other cols in this data_source"""
        raise NotImplementedError

    def dynamic_features(self) -> list:
        """the dynamic features in this data_source"""
        if hasattr(self, "aqua_fetch"):
            original_features = self.aqua_fetch.dynamic_features
            return self._clean_feature_names(original_features)
        raise NotImplementedError

    @staticmethod
    def _clean_feature_names(feature_names):
        """
        Clean feature names to be compatible with NetCDF format.

        Parameters
        ----------
        feature_names : list or pd.Index
            Original feature names

        Returns
        -------
        list
            Cleaned feature names
        """
        import pandas as pd

        if isinstance(feature_names, list):
            feature_names = pd.Index(feature_names)
        return (
            feature_names.str.strip().str.replace(" ", "_").str.replace("/", "_")
        ).tolist()

    def static_features(self) -> list:
        """the static features in this data_source"""
        if hasattr(self, "aqua_fetch"):
            original_features = self.aqua_fetch.static_features
            return self._clean_feature_names(original_features)
        raise NotImplementedError

    @property
    @abstractmethod
    def _attributes_cache_filename(self):
        pass

    @property
    @abstractmethod
    def _timeseries_cache_filename(self):
        pass

    @property
    @abstractmethod
    def default_t_range(self):
        pass

    @abstractmethod
    @abstractmethod
    def _get_timeseries_units(self) -> list:
        raise NotImplementedError

    def cache_timeseries_xrdataset(self):
        if hasattr(self, "aqua_fetch"):
            gage_id_lst = self.read_object_ids().tolist()
            # Get original variable names from aqua_fetch
            original_var_lst = self.aqua_fetch.dynamic_features
            # Get cleaned variable names
            cleaned_var_lst = self.dynamic_features()
            units = self._get_timeseries_units()

            batch_data = self.aqua_fetch.fetch_stations_features(
                stations=gage_id_lst,
                dynamic_features=original_var_lst,
                static_features=None,
                st=self.default_t_range[0],
                en=self.default_t_range[1],
                as_dataframe=False,
            )

            dynamic_data = (
                batch_data[1] if isinstance(batch_data, tuple) else batch_data
            )

            new_data_vars = {}
            time_coord = dynamic_data.coords["time"]

            for var_idx, (original_var, cleaned_var) in enumerate(
                tqdm(
                    zip(original_var_lst, cleaned_var_lst),
                    desc="Processing variables",
                    total=len(original_var_lst),
                )
            ):
                var_data = []
                for station in gage_id_lst:
                    if station in dynamic_data.data_vars:
                        station_data = dynamic_data[station].sel(
                            dynamic_features=original_var
                        )
                        if "dynamic_features" in station_data.coords:
                            station_data = station_data.drop("dynamic_features")
                        var_data.append(station_data)

                if var_data:
                    combined = xr.concat(var_data, dim="basin")
                    combined["basin"] = gage_id_lst
                    combined.attrs["units"] = (
                        units[var_idx] if var_idx < len(units) else "unknown"
                    )
                    # Use cleaned variable name as the key
                    new_data_vars[cleaned_var] = combined

            new_ds = xr.Dataset(
                data_vars=new_data_vars,
                coords={
                    "basin": gage_id_lst,
                    "time": time_coord,
                },
            )

            batch_filepath = self.cache_dir.joinpath(self._timeseries_cache_filename)
            batch_filepath.parent.mkdir(parents=True, exist_ok=True)
            new_ds.to_netcdf(batch_filepath)
            print(f"成功保存到: {batch_filepath}")
        else:
            raise NotImplementedError

    def _assign_units_to_dataset(self, ds, units_map):
        def get_unit_by_prefix(var_name):
            for prefix, unit in units_map.items():
                if var_name.startswith(prefix):
                    return unit
            return None

        def get_unit(var_name):
            prefix_unit = get_unit_by_prefix(var_name)
            if prefix_unit:
                return prefix_unit
            return "undefined"

        for var in ds.data_vars:
            unit = get_unit(var)
            ds[var].attrs["units"] = unit
            if unit == "class":
                ds[var].attrs["description"] = "Classification code"
        return ds

    @abstractmethod
    def _get_attribute_units(self) -> dict:
        raise NotImplementedError

    def cache_attributes_xrdataset(self):
        if hasattr(self, "aqua_fetch"):
            df_attr = self.aqua_fetch.fetch_static_features()
            print(df_attr.columns)
            # Clean column names using the unified method
            df_attr.columns = self._clean_feature_names(df_attr.columns)
            # Remove duplicate columns if any (keep first occurrence)
            if df_attr.columns.duplicated().any():
                df_attr = df_attr.loc[:, ~df_attr.columns.duplicated()]
            # Ensure index is string type for basin IDs
            df_attr.index = df_attr.index.astype(str)
            ds_attr = df_attr.to_xarray()
            # Check if the coordinate is named 'basin', if not rename it
            coord_names = list(ds_attr.dims.keys())
            if len(coord_names) > 0 and coord_names[0] != "basin":
                ds_attr = ds_attr.rename({coord_names[0]: "basin"})
            units_map = self._get_attribute_units()
            ds_attr = self._assign_units_to_dataset(ds_attr, units_map)
            ds_attr.to_netcdf(self.cache_dir.joinpath(self._attributes_cache_filename))
        else:
            raise NotImplementedError

    def read_attr_xrdataset(self, gage_id_lst=None, var_lst=None, **kwargs):
        attr_cache_file = self.cache_dir.joinpath(self._attributes_cache_filename)
        try:
            attr = xr.open_dataset(attr_cache_file)
        except FileNotFoundError:
            self.cache_attributes_xrdataset()
            attr = xr.open_dataset(attr_cache_file)
        if var_lst is None or len(var_lst) == 0:
            var_lst = self.static_features()
        # Ensure gage_id_lst is string type to match basin coordinate
        if gage_id_lst is not None:
            gage_id_lst = [str(gid) for gid in gage_id_lst]
        return attr[var_lst].sel(basin=gage_id_lst)

    def read_ts_xrdataset(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs,
    ):
        if var_lst is None:
            var_lst = self.dynamic_features()
        if t_range is None:
            t_range = self.default_t_range
        ts_cache_file = self.cache_dir.joinpath(self._timeseries_cache_filename)
        if not os.path.isfile(ts_cache_file):
            self.cache_timeseries_xrdataset()
        ts = xr.open_dataset(ts_cache_file)
        all_vars = ts.data_vars
        if any(var not in ts.variables for var in var_lst):
            raise ValueError(f"var_lst must all be in {all_vars}")
        return ts[var_lst].sel(basin=gage_id_lst, time=slice(t_range[0], t_range[1]))

    def read_area(self, gage_id_lst):
        """read area of each basin/unit"""
        raise NotImplementedError

    def read_mean_prcp(self, gage_id_lst, unit="mm/d"):
        """read mean precipitation of each basin
        default unit is mm/d, but one can chose other units and we will convert the unit to the specified unit

        Parameters
        ----------
        gage_id_lst : list, optional
            the list of gage ids, by default None
        unit : str, optional
            the unit of precipitation, by default "mm/d"

        Returns
        -------
        xr.Dataset
            the mean precipitation of each basin
        """
        raise NotImplementedError
