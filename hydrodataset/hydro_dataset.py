"""
Author: Wenyu Ouyang
Date: 2022-09-05 23:20:24
LastEditTime: 2025-11-06 19:26:20
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
import pandas as pd
import numpy as np

from hydrodataset import ROOT_DIR, CACHE_DIR


class StandardVariable:
    """A class to hold standardized variable names as constants."""

    STREAMFLOW = "streamflow"
    WATER_LEVEL = "water_level"

    PRECIPITATION = "precipitation"
    CRAINF_FRAC = "crainf_frac"  # Fraction of total precipitation that is convective
    PRECIPITATION_MIN = "precipitation_min"
    PRECIPITATION_MAX = "precipitation_max"
    PRECIPITATION_MEDIAN = "precipitation_median"

    TEMPERATURE_MAX = "temperature_max"
    TEMPERATURE_MIN = "temperature_min"
    TEMPERATURE_MEAN = "temperature_mean"

    DAYLIGHT_DURATION = "daylight_duration"
    RELATIVE_DAYLIGHT_DURATION = "relative_daylight_duration"

    SOLAR_RADIATION = "solar_radiation"
    SOLAR_RADIATION_MIN = "solar_radiation_min"
    SOLAR_RADIATION_MAX = "solar_radiation_max"
    SOLAR_RADIATION_MEDIAN = "solar_radiation_median"
    THERMAL_RADIATION = "thermal_radiation"
    THERMAL_RADIATION_MIN = "thermal_radiation_min"
    THERMAL_RADIATION_MAX = "thermal_radiation_max"
    LONGWAVE_SOLAR_RADIATION = "longwave_solar_radiation"

    SNOW_WATER_EQUIVALENT = "snow_water_equivalent"
    SNOW_WATER_EQUIVALENT_MIN = "snow_water_equivalent_min"
    SNOW_WATER_EQUIVALENT_MAX = "snow_water_equivalent_max"
    SNOW_DEPTH = "snow_depth"
    SNOW_COVER = "snow_cover"
    SNOW_SUBLIMATION = "snow_sublimation"
    SNOW_DENSITY = "snow_density"

    VAPOR_PRESSURE = "vapor_pressure"

    SURFACE_PRESSURE = "surface_pressure"
    SURFACE_PRESSURE_MIN = "surface_pressure_min"
    SURFACE_PRESSURE_MAX = "surface_pressure_max"

    WIND_SPEED = "wind_speed"
    U_WIND_SPEED = "u_wind_speed"
    U_WIND_SPEED_MIN = "u_wind_speed_min"
    U_WIND_SPEED_MAX = "u_wind_speed_max"
    V_WIND_SPEED = "v_wind_speed"
    V_WIND_SPEED_MIN = "v_wind_speed_min"
    V_WIND_SPEED_MAX = "v_wind_speed_max"
    WIND_DIR = "wind_dir"
    LOW_LEVEL_WIND_SHEAR = "low_level_wind_shear"
    DEEP_LEVEL_WIND_SHEAR = "deep_level_wind_shear"

    RELATIVE_HUMIDITY = "relative_humidity"
    SPECIFIC_HUMIDITY = "specific_humidity"
    RELATIVE_HUMIDITY_MIN = "relative_humidity_min"
    RELATIVE_HUMIDITY_MAX = "relative_humidity_max"
    RELATIVE_HUMIDITY_MEDIAN = "relative_humidity_median"
    TOTAL_COLUMN_WATER_VAPOUR = "total_column_water_vapour"

    CAPE = "cape"  # Convective available potential energy
    CIN = "cin"  # Convective inhibition

    POTENTIAL_EVAPOTRANSPIRATION = "potential_evapotranspiration"
    EVAPORATION = "evaporation"
    EVAPOTRANSPIRATION = "evapotranspiration"

    SOIL_MOISTURE = "soil_moisture"
    VOLUMETRIC_SOIL_WATER_LAYER1 = "volumetric_soil_water_layer1"  # 0-7cm
    VOLUMETRIC_SOIL_WATER_LAYER1_MIN = "volumetric_soil_water_layer1_min"
    VOLUMETRIC_SOIL_WATER_LAYER1_MAX = "volumetric_soil_water_layer1_max"
    VOLUMETRIC_SOIL_WATER_LAYER2 = "volumetric_soil_water_layer2"  # 7-28cm
    VOLUMETRIC_SOIL_WATER_LAYER2_MIN = "volumetric_soil_water_layer2_min"
    VOLUMETRIC_SOIL_WATER_LAYER2_MAX = "volumetric_soil_water_layer2_max"
    VOLUMETRIC_SOIL_WATER_LAYER3 = "volumetric_soil_water_layer3"  # 28-100cm
    VOLUMETRIC_SOIL_WATER_LAYER3_MIN = "volumetric_soil_water_layer3_min"
    VOLUMETRIC_SOIL_WATER_LAYER3_MAX = "volumetric_soil_water_layer3_max"
    VOLUMETRIC_SOIL_WATER_LAYER4 = "volumetric_soil_water_layer4"  # 100-289cm
    VOLUMETRIC_SOIL_WATER_LAYER4_MIN = "volumetric_soil_water_layer4_min"
    VOLUMETRIC_SOIL_WATER_LAYER4_MAX = "volumetric_soil_water_layer4_max"

    MIN_RAIN_RATE = "min_rain_rate"
    MAX_RAIN_RATE = "max_rain_rate"

    GROUND_HEAT_FLUX = "ground_heat_flux"


class HydroDataset(ABC):
    """An interface for Hydrological Dataset

    For unit, we use Pint package's unit system -- unit registry

    Parameters
    ----------
    ABC : _type_
        _description_
    """

    # A unified definition for static variables, including name mapping and units
    _base_static_definitions = {
        "area": {"specific_name": "area_km2", "unit": "km^2"},
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
    }
    # variable name map for timeseries
    _dynamic_variable_mapping = {}

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

        # Merge static variable definitions
        self._static_variable_definitions = self._base_static_definitions.copy()
        if hasattr(self.__class__, "_subclass_static_definitions"):
            self._static_variable_definitions.update(self._subclass_static_definitions)

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

    def _dynamic_features(self) -> list:
        """the dynamic features in this data_source"""
        if hasattr(self, "aqua_fetch"):
            original_features = self.aqua_fetch.dynamic_features
            return self._clean_feature_names(original_features)
        raise NotImplementedError

    @staticmethod
    def _clean_feature_names(feature_names):
        """Clean feature names to be compatible with NetCDF format and our internal standard.

        The cleaning process follows these steps:
        1. Remove units in parentheses (along with any preceding whitespace)
           e.g., 'Prcp(mm/day)' -> 'Prcp' or 'Temp (°C)' -> 'Temp'
        2. Convert all characters to lowercase
           e.g., 'Prcp' -> 'prcp'
        3. Remove any remaining invalid characters (only keep a-z, 0-9, and _)
           This ensures NetCDF variable naming compliance

        Args:
            feature_names (list or pd.Index): Original feature names that may contain
                units and special characters

        Returns:
            list: Cleaned feature names with only lowercase letters, numbers, and underscores

        Examples:
            >>> _clean_feature_names(['Prcp(mm/day)_daymet', 'Temp (°C)'])
            ['prcp_daymet', 'temp']
        """
        if not isinstance(feature_names, pd.Index):
            feature_names = pd.Index(feature_names)

        # Remove units in parentheses, then convert to lowercase
        cleaned_names = feature_names.str.replace(
            r"\s*\([^)]*\)", "", regex=True
        ).str.lower()
        # Replace any remaining invalid characters
        cleaned_names = cleaned_names.str.replace(r"""[^a-z0-9_]""", "", regex=True)
        return cleaned_names.tolist()

    def _static_features(self) -> list:
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

    def cache_timeseries_xrdataset(self):
        if hasattr(self, "aqua_fetch"):
            # Build a lookup map from specific name to unit
            unit_lookup = {}
            if hasattr(self, "_dynamic_variable_mapping"):
                for (
                    std_name,
                    mapping_info,
                ) in self._dynamic_variable_mapping.items():
                    for source, source_info in mapping_info["sources"].items():
                        unit_lookup[source_info["specific_name"]] = source_info["unit"]

            gage_id_lst = self.read_object_ids().tolist()
            original_var_lst = self.aqua_fetch.dynamic_features
            cleaned_var_lst = self._clean_feature_names(original_var_lst)
            # Create a mapping from original variable names to cleaned names
            # to ensure correct correspondence even if list order changes
            var_name_mapping = dict(zip(original_var_lst, cleaned_var_lst))

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

            # Process only the variables that exist in the data source
            # Subclasses can add additional variables in their override methods
            for original_var in tqdm(
                original_var_lst,
                desc="Processing variables",
                total=len(original_var_lst),
            ):
                cleaned_var = var_name_mapping[original_var]
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
                    combined.attrs["units"] = unit_lookup.get(cleaned_var, "unknown")
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

        return ds

    def _get_attribute_units(self) -> dict:
        """Builds a unit dictionary from the static variable definitions."""
        return {
            info["specific_name"]: info["unit"]
            for std_name, info in self._static_variable_definitions.items()
        }

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

    def read_attr_xrdataset(
        self,
        gage_id_lst: list = None,
        var_lst: list = None,
        to_numeric: bool = True,
        **kwargs,
    ) -> xr.Dataset:
        """Reads attribute data for a list of basins using standardized variable names.

        Args:
            gage_id_lst: A list of basin identifiers.
            var_lst: A list of **standard** attribute names to retrieve.
                If None, all available static features will be returned.
            to_numeric: If True, converts all non-numeric variables to numeric codes
                and stores the original labels in the variable's attributes.
                Defaults to True.

        Returns:
            An xarray Dataset containing the attribute data for the requested basins,
            with variables named using the standard names.
        """
        if var_lst is None:
            var_lst = self.get_available_static_features()

        # 1. Translate standard names to dataset-specific names
        target_vars_to_fetch = []
        rename_map = {}
        for std_name in var_lst:
            if std_name not in self._static_variable_definitions:
                raise ValueError(
                    f"'{std_name}' is not a recognized standard static variable."
                )
            actual_var_name = self._static_variable_definitions[std_name][
                "specific_name"
            ]
            target_vars_to_fetch.append(actual_var_name)
            rename_map[actual_var_name] = std_name

        # 2. Read data from cache using actual variable names
        attr_cache_file = self.cache_dir.joinpath(self._attributes_cache_filename)
        try:
            attr_ds = xr.open_dataset(attr_cache_file)
        except FileNotFoundError:
            self.cache_attributes_xrdataset()
            attr_ds = xr.open_dataset(attr_cache_file)

        # 3. Select variables and basins
        ds_subset = attr_ds[target_vars_to_fetch]
        if gage_id_lst is not None:
            gage_id_lst = [str(gid) for gid in gage_id_lst]
            ds_selected = ds_subset.sel(basin=gage_id_lst)
        else:
            ds_selected = ds_subset

        # 4. Rename to standard names
        final_ds = ds_selected.rename(rename_map)

        if not to_numeric:
            return final_ds

        # 5. If to_numeric is True, perform conversion
        converted_ds = xr.Dataset(coords=final_ds.coords)
        for var_name, da in final_ds.data_vars.items():
            if np.issubdtype(da.dtype, np.number):
                converted_ds[var_name] = da
            else:
                # Assumes string-like array that needs factorizing
                numeric_vals, labels = pd.factorize(da.values, sort=True)
                new_da = xr.DataArray(
                    numeric_vals,
                    coords=da.coords,
                    dims=da.dims,
                    name=da.name,
                    attrs=da.attrs,  # Preserve original attributes
                )
                new_da.attrs["labels"] = labels.tolist()
                converted_ds[var_name] = new_da
        return converted_ds

    def _load_ts_dataset(self, **kwargs):
        """
        Loads the time series dataset from cache.

        This method can be overridden by subclasses to implement different loading
        strategies (e.g., loading multiple files).

        Args:
            **kwargs: Additional keyword arguments for loading.

        Returns:
            xarray.Dataset: The loaded time series dataset.
        """
        ts_cache_file = self.cache_dir.joinpath(self._timeseries_cache_filename)

        if not os.path.isfile(ts_cache_file):
            self.cache_timeseries_xrdataset()

        return xr.open_dataset(ts_cache_file)

    def read_ts_xrdataset(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        sources: dict = None,
        **kwargs,
    ):
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

        for std_name in var_lst:
            if std_name not in self._dynamic_variable_mapping:
                raise ValueError(
                    f"'{std_name}' is not a recognized standard variable for this dataset."
                )

            mapping_info = self._dynamic_variable_mapping[std_name]

            # Determine which source(s) to use and if they were explicitly requested
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

            # A suffix is only needed if the user explicitly requested multiple sources
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

        # Read data from cache using actual variable names
        ts = self._load_ts_dataset(**kwargs)
        missing_vars = [v for v in target_vars_to_fetch if v not in ts.data_vars]
        if missing_vars:
            # To provide a better error message, map back to standard names
            reverse_rename_map = {v: k for k, v in rename_map.items()}
            missing_std_vars = [reverse_rename_map.get(v, v) for v in missing_vars]
            raise ValueError(
                f"The following variables are missing from the cache file: {missing_std_vars}"
            )

        ds_subset = ts[target_vars_to_fetch]
        ds_selected = ds_subset.sel(
            basin=gage_id_lst, time=slice(t_range[0], t_range[1])
        )
        final_ds = ds_selected.rename(rename_map)
        return final_ds

    def get_available_dynamic_features(self) -> dict:
        """
        Returns a dictionary of available standard dynamic feature names
        and their possible sources.
        """
        if (
            not hasattr(self, "_dynamic_variable_mapping")
            or not self._dynamic_variable_mapping
        ):
            return {}

        feature_info = {}
        for std_name, mapping_info in self._dynamic_variable_mapping.items():
            feature_info[std_name] = {
                "default_source": mapping_info.get("default_source"),
                "available_sources": list(mapping_info.get("sources", {}).keys()),
            }
        return feature_info

    def get_available_static_features(self) -> list:
        """Returns a list of available standard static feature names."""
        return list(self._static_variable_definitions.keys())

    @property
    def available_static_features(self) -> list:
        """Returns a list of available static attribute names."""
        return self.get_available_static_features()

    @property
    def available_dynamic_features(self) -> dict:
        """Returns a dictionary of available dynamic feature names and their possible sources."""
        return self.get_available_dynamic_features()

    def read_area(self, gage_id_lst: list[str]) -> xr.Dataset:
        """Reads the catchment area for a list of basins.

        Args:
            gage_id_lst: A list of basin identifiers for which to retrieve the area.

        Returns:
            An xarray Dataset containing the area data for the requested basins.
        """
        data_ds = self.read_attr_xrdataset(gage_id_lst=gage_id_lst, var_lst=["area"])
        return data_ds

    def read_mean_prcp(self, gage_id_lst: list[str], unit: str = "mm/d") -> xr.Dataset:
        """Reads the mean daily precipitation for a list of basins, with unit conversion.

        Args:
            gage_id_lst: A list of basin identifiers.
            unit: The desired unit for the output precipitation. Defaults to "mm/d".
                Supported units: ['mm/d', 'mm/day', 'mm/h', 'mm/hour', 'mm/3h',
                'mm/3hour', 'mm/8d', 'mm/8day'].

        Returns:
            An xarray Dataset containing the mean precipitation data in the specified units.

        Raises:
            ValueError: If an unsupported unit is provided.
        """
        prcp_var_name = "p_mean"
        data_ds = self.read_attr_xrdataset(
            gage_id_lst=gage_id_lst, var_lst=[prcp_var_name]
        )
        # No conversion needed
        if unit in ["mm/d", "mm/day"]:
            return data_ds

        # Conversion needed, create a new dataset
        converted_ds = data_ds.copy()
        # After renaming, the variable in the dataset is now the standard name
        if unit in ["mm/h", "mm/hour"]:
            converted_ds[prcp_var_name] = data_ds[prcp_var_name] / 24
        elif unit in ["mm/3h", "mm/3hour"]:
            converted_ds[prcp_var_name] = data_ds[prcp_var_name] / 8
        elif unit in ["mm/8d", "mm/8day"]:
            converted_ds[prcp_var_name] = data_ds[prcp_var_name] * 8
        else:
            raise ValueError(
                "unit must be one of ['mm/d', 'mm/day', 'mm/h', 'mm/hour', 'mm/3h', 'mm/3hour', 'mm/8d', 'mm/8day']"
            )
        return converted_ds
