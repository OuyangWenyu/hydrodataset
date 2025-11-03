"""
Author: Wenyu Ouyang
Date: 2025-10-27 14:52:23
LastEditTime: 2025-10-28 20:13:06
LastEditors: Wenyu Ouyang
Description: CAMELS-BR dataset class.
FilePath: \hydrodataset\hydrodataset\camels_br.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import collections
import os
import logging
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
from aqua_fetch import CAMELS_BR
from hydrodataset import HydroDataset, StandardVariable


class CamelsBr(HydroDataset):
    """CAMELS_BR dataset class.

    This class uses a custom data reading implementation to support a newer
    dataset version than the one supported by the underlying aquafetch library.
    It overrides the download URLs and provides its own parsing and caching logic.
    """

    def __init__(self, data_path, region=None, download=False):
        """Initialize CAMELS_BR dataset.

        Args:
            data_path: Path to the CAMELS_BR data directory.
            region: Geographic region identifier (optional, defaults to BR).
            download: Whether to download data automatically (not used, handled by aqua_fetch).
        """
        super().__init__(data_path)
        self.region = "BR" if region is None else region

        # Define the new URLs for the latest dataset version
        new_url = "https://zenodo.org/records/15025488"
        new_urls = {
            "01_CAMELS_BR_attributes.zip": "https://zenodo.org/records/15025488/files/",
            "02_CAMELS_BR_streamflow_all_catchments.zip": "https://zenodo.org/records/15025488/files/",
            "03_CAMELS_BR_streamflow_selected_catchments.zip": "https://zenodo.org/records/15025488/files/",
            "04_CAMELS_BR_streamflow_simulated.zip": "https://zenodo.org/records/15025488/files/",
            "05_CAMELS_BR_precipitation.zip": "https://zenodo.org/records/15025488/files/",
            "06_CAMELS_BR_actual_evapotransp.zip": "https://zenodo.org/records/15025488/files/",
            "07_CAMELS_BR_potential_evapotransp.zip": "https://zenodo.org/records/15025488/files/",
            "08_CAMELS_BR_reference_evapotransp.zip": "https://zenodo.org/records/15025488/files/",
            "09_CAMELS_BR_temperature.zip": "https://zenodo.org/records/15025488/files/",
            "10_CAMELS_BR_soil_moisture.zip": "https://zenodo.org/records/15025488/files/",
            "11_CAMELS_BR_precipitation_ana_gauges.zip": "https://zenodo.org/records/15025488/files/",
            "12_CAMELS_BR_catchment_boundaries.zip": "https://zenodo.org/records/15025488/files/",
            "13_CAMELS_BR_gauge_location.zip": "https://zenodo.org/records/15025488/files/",
            "CAMELS_BR_readme.txt": "https://zenodo.org/records/15025488/files/",
        }
        new_folders = {
            "streamflow_mm": "03_CAMELS_BR_streamflow_selected_catchments",
        }

        def do_nothing(self, *args, **kwargs):
            pass

        class_attrs = {
            "url": new_url,
            "urls": new_urls,
            "folders": new_folders,
            "_maybe_to_netcdf": do_nothing,
        }
        CustomCamelsBr = type("CAMELS_BR", (CAMELS_BR,), class_attrs)

        # Instantiate our custom class to handle downloads, but note that the reading
        # logic below is custom and does not rely on aquafetch's parsing.
        self.aqua_fetch = CustomCamelsBr(data_path)

        self.data_source_description = self.set_data_source_describe()

        # Build the variable map for the custom reading logic
        self._variable_map = self._build_variable_map()

    @property
    def _attributes_cache_filename(self):
        return "camels_br_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camels_br_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1980-01-01", "2024-07-31"]

    # get the information of features from dataset file"CAMELS_BR_readme"
    _subclass_static_definitions = {
        "area": {"specific_name": "area_km2", "unit": "km^2"},
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
    }

    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "m3s",
            "sources": {
                "m3s": {"specific_name": "streamflow_m3s", "unit": "m^3/s"},
                "mm": {"specific_name": "streamflow_mm", "unit": "mm/day"},
                "simulated": {
                    "specific_name": "simulated_streamflow_m3s",
                    "unit": "m^3/s",
                },
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "era5land",
            "sources": {
                "era5land": {"specific_name": "p_era5land", "unit": "mm/day"},
                "mswep": {"specific_name": "p_mswep", "unit": "mm/day"},
                "cpc": {"specific_name": "p_cpc", "unit": "mm/day"},
                "chirps": {"specific_name": "p_chirps", "unit": "mm/day"},
                "brdwgd": {"specific_name": "p_brdwgd", "unit": "mm/day"},
                "ana_gauges": {"specific_name": "p_ana_gauges", "unit": "mm/day"},
            },
        },
        StandardVariable.EVAPOTRANSPIRATION: {
            "default_source": "era5land",
            "sources": {
                "era5land": {"specific_name": "aet_era5land", "unit": "mm/day"},
                "gleam": {"specific_name": "aet_gleam", "unit": "mm/day"},
                "mgb": {"specific_name": "aet_mgb", "unit": "mm/day"},
            },
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "era5land",
            "sources": {
                "era5land": {"specific_name": "pet_era5land", "unit": "mm/day"},
                "gleam": {"specific_name": "pet_gleam", "unit": "mm/day"},
            },
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "era5land",
            "sources": {
                "era5land": {"specific_name": "tmax_era5land", "unit": "°C"},
                "cpc": {"specific_name": "tmax_cpc", "unit": "°C"},
                "brdwgd": {"specific_name": "tmax_brdwgd", "unit": "°C"},
            },
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "era5land",
            "sources": {
                "era5land": {"specific_name": "tmin_era5land", "unit": "°C"},
                "cpc": {"specific_name": "tmin_cpc", "unit": "°C"},
                "brdwgd": {"specific_name": "tmin_brdwgd", "unit": "°C"},
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "era5land",
            "sources": {
                "era5land": {"specific_name": "tmean_era5land", "unit": "°C"},
            },
        },
        StandardVariable.SOIL_MOISTURE: {
            "default_source": "surface_gleam",
            "sources": {
                "surface_gleam": {
                    "specific_name": "sm_surface_gleam",
                    "unit": "m^3/m^3",
                },
                "rootzone_gleam": {
                    "specific_name": "sm_rootzone_gleam",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER1: {
            "default_source": "era5land",
            "sources": {
                "era5land": {"specific_name": "sm_layer1_era5land", "unit": "m^3/m^3"},
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER2: {
            "default_source": "era5land",
            "sources": {
                "era5land": {"specific_name": "sm_layer2_era5land", "unit": "m^3/m^3"},
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER3: {
            "default_source": "era5land",
            "sources": {
                "era5land": {"specific_name": "sm_layer3_era5land", "unit": "m^3/m^3"},
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER4: {
            "default_source": "era5land",
            "sources": {
                "era5land": {"specific_name": "sm_layer4_era5land", "unit": "m^3/m^3"},
            },
        },
    }

    def _build_variable_map(self):
        """
        Scans all time-series directories to build a map from each variable
        to its parent directory path. This is done once at initialization.
        """
        variable_map = {}
        all_ts_dirs = (
            self.data_source_description["CAMELS_FORCING_DIR"]
            + self.data_source_description["CAMELS_FLOW_DIR"]
        )

        try:
            sample_gage_id = self.read_object_ids()[0]
        except IndexError:
            # If there are no gages, we can't build the map.
            return {}

        for ts_dir in all_ts_dirs:
            base_name = str(ts_dir).split(os.sep)[-1][13:]
            # Handle special case for precipitation_ana_gauges
            if base_name == "precipitation_ana_gauges":
                variable_map["p_ana_gauges"] = str(ts_dir)
                continue

            # Find a sample file to read its header
            try:
                files_for_gage = [
                    f for f in os.listdir(ts_dir) if f.startswith(sample_gage_id)
                ]
                if not files_for_gage:
                    continue
                sample_file_path = os.path.join(ts_dir, files_for_gage[0])
                df_header = pd.read_csv(sample_file_path, sep=r"\s+", nrows=0)
                internal_vars = df_header.columns[3:]
                for var in internal_vars:
                    if var in variable_map:
                        logging.warning(
                            f"Duplicate variable '{var}' found. Overwriting mapping."
                        )
                    variable_map[var] = str(ts_dir)
            except (FileNotFoundError, IndexError, pd.errors.EmptyDataError):
                # If we can't read a sample file, just skip this directory
                logging.warning(
                    f"Could not read sample file in {ts_dir} to map variables."
                )
                continue
        return variable_map

    def set_data_source_describe(self) -> collections.OrderedDict:
        """
        the files in the dataset and their location in file system

        Returns
        -------
        collections.OrderedDict
            the description for a CAMELS-BR dataset
        """
        camels_db = self.data_source_dir.joinpath("CAMELS_BR")

        # attr
        attr_dir = camels_db.joinpath(
            "01_CAMELS_BR_attributes", "01_CAMELS_BR_attributes"
        )
        # we don't need the location attr file
        attr_key_lst = [
            "climate",
            "geology",
            "human_intervention",
            "hydrology",
            "land_cover",
            "quality_check",
            "soil",
            "topography",
        ]
        # id and name, there are two types stations in CAMELS_BR, and we only chose the 897-stations version
        gauge_id_file = attr_dir.joinpath("camels_br_topography.txt")
        # shp file of basins
        camels_shp_file = camels_db.joinpath(
            "12_CAMELS_BR_catchment_boundaries",
            "12_CAMELS_BR_catchment_boundaries",
            "camels_br_catchments.gpkg",
        )
        # config of flow data
        flow_dir = camels_db.joinpath(
            "03_CAMELS_BR_streamflow_selected_catchments",
            "03_CAMELS_BR_streamflow_selected_catchments",
        )
        flow_dir_simulated = camels_db.joinpath(
            "04_CAMELS_BR_streamflow_simulated",
            "04_CAMELS_BR_streamflow_simulated",
        )

        # forcing
        forcing_dir_precipitation = camels_db.joinpath(
            "05_CAMELS_BR_precipitation",
            "05_CAMELS_BR_precipitation",
        )
        forcing_dir_evapotransp = camels_db.joinpath(
            "06_CAMELS_BR_actual_evapotransp",
            "06_CAMELS_BR_actual_evapotransp",
        )
        forcing_dir_potential_evapotransp = camels_db.joinpath(
            "07_CAMELS_BR_potential_evapotransp",
            "07_CAMELS_BR_potential_evapotransp",
        )
        forcing_dir_reference_evap = camels_db.joinpath(
            "08_CAMELS_BR_reference_evapotransp",
            "08_CAMELS_BR_reference_evapotransp",
        )
        forcing_dir_temperature = camels_db.joinpath(
            "09_CAMELS_BR_temperature",
            "09_CAMELS_BR_temperature",
        )
        forcing_dir_soilmoisture = camels_db.joinpath(
            "10_CAMELS_BR_soil_moisture",
            "10_CAMELS_BR_soil_moisture",
        )
        forcing_dir_precipitation_ana_gauges = camels_db.joinpath(
            "11_CAMELS_BR_precipitation_ana_gauges",
            "11_CAMELS_BR_precipitation_ana_gauges",
        )
        base_url = "https://zenodo.org/records/15025488"
        # NOTE: Now the CAMELS_BR is not supported by AquaFetch,
        # Here, we only add download urls to be used for unzipping the dataset.
        download_url_lst = [
            f"{base_url}/files/01_CAMELS_BR_attributes.zip",
            f"{base_url}/files/02_CAMELS_BR_streamflow_all_catchments.zip",
            f"{base_url}/files/03_CAMELS_BR_streamflow_selected_catchments.zip",
            f"{base_url}/files/04_CAMELS_BR_streamflow_simulated.zip",
            f"{base_url}/files/05_CAMELS_BR_precipitation.zip",
            f"{base_url}/files/06_CAMELS_BR_actual_evapotransp.zip",
            f"{base_url}/files/07_CAMELS_BR_potential_evapotransp.zip",
            f"{base_url}/files/08_CAMELS_BR_reference_evapotransp.zip",
            f"{base_url}/files/09_CAMELS_BR_temperature.zip",
            f"{base_url}/files/10_CAMELS_BR_soil_moisture.zip",
            f"{base_url}/files/11_CAMELS_BR_precipitation_ana_gauges.zip",
            f"{base_url}/files/12_CAMELS_BR_catchment_boundaries.zip",
            f"{base_url}/files/13_CAMELS_BR_gauge_location.zip",
            f"{base_url}/files/CAMELS_BR_readme.txt",
        ]
        return collections.OrderedDict(
            CAMELS_DIR=camels_db,
            CAMELS_FLOW_DIR=[
                flow_dir,
                flow_dir_simulated,
            ],
            CAMELS_FORCING_DIR=[
                forcing_dir_precipitation,
                forcing_dir_precipitation_ana_gauges,
                forcing_dir_evapotransp,
                forcing_dir_potential_evapotransp,
                forcing_dir_reference_evap,
                forcing_dir_temperature,
                forcing_dir_soilmoisture,
            ],
            CAMELS_ATTR_DIR=attr_dir,
            CAMELS_ATTR_KEY_LST=attr_key_lst,
            CAMELS_GAUGE_FILE=gauge_id_file,
            CAMELS_BASINS_SHP_FILE=camels_shp_file,
            CAMELS_DOWNLOAD_URL_LST=download_url_lst,
        )

    def _get_constant_cols_some(self, data_folder, prefix, postfix, sep):
        var_dict = {}
        var_lst = []
        key_lst = self.data_source_description["CAMELS_ATTR_KEY_LST"]
        for key in key_lst:
            data_file = os.path.join(data_folder, prefix + key + postfix)
            data_temp = pd.read_csv(data_file, sep=sep)
            var_lst_temp = list(data_temp.columns[1:])
            var_dict[key] = var_lst_temp
            var_lst.extend(var_lst_temp)
        return np.array(var_lst)

    def _static_features(self) -> list:
        """
        all readable attrs in CAMELS-BR

        Returns
        -------
        list
            attribute types
        """
        data_folder = self.data_source_description["CAMELS_ATTR_DIR"]
        return self._get_constant_cols_some(data_folder, "camels_br_", ".txt", "\s+")

    def _dynamic_features(self):
        "Return all available time series variables."
        return np.array(list(self._variable_map.keys()))

    def _find_file_for_gage(self, directory, gage_id):
        """Finds the data file for a specific gage in a given directory."""
        if not os.path.isdir(directory):
            return None
        # Find any file in the directory for our sample gage
        gage_files = [f for f in os.listdir(directory) if f.startswith(gage_id)]
        if not gage_files:
            return None
        return os.path.join(directory, gage_files[0])

    def read_relevant_cols(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        forcing_type: str = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Read time series data for a list of variables, optimizing I/O by grouping variables by file.

        Parameters
        ----------
        gage_id_lst
            station ids
        t_range
            the time range, for example, ["1990-01-01", "2000-01-01"]
        var_lst
            time series variable types (e.g., ["p_chirps", "t_mean"])
        Returns
        -------
        np.array
            time series data
        """
        if var_lst is None or len(var_lst) == 0:
            return np.array([])
        t_range_list = pd.date_range(start=t_range[0], end=t_range[1], freq="D").values
        nt = t_range_list.shape[0]
        x = np.full([len(gage_id_lst), nt, len(var_lst)], np.nan)

        for k, gage_id in enumerate(tqdm(gage_id_lst, desc="Reading basins")):
            # Group variables by the directory they belong to for the current basin
            dir_to_vars_map = {}
            for i, var in enumerate(var_lst):
                directory = self._variable_map.get(var)
                if not directory:
                    logging.warning(f"Could not find directory for variable: {var}")
                    continue
                if directory not in dir_to_vars_map:
                    dir_to_vars_map[directory] = []
                dir_to_vars_map[directory].append((var, i))

            # For this basin, iterate through directories, reading each file only once
            for directory, vars_in_dir in dir_to_vars_map.items():
                file_path = self._find_file_for_gage(directory, gage_id)
                if not file_path:
                    logging.warning(f"No file found for gage {gage_id} in {directory}")
                    continue

                try:
                    data_temp = pd.read_csv(file_path, sep=r"\s+")
                except (FileNotFoundError, pd.errors.EmptyDataError):
                    logging.warning(f"Could not read or empty file: {file_path}")
                    continue

                # Intersect time once per file
                df_date = data_temp[["year", "month", "day"]]
                date = pd.to_datetime(df_date).values.astype("datetime64[D]")
                [c, file_indices, target_indices] = np.intersect1d(
                    date, t_range_list, return_indices=True
                )

                # For each variable belonging to this file, extract its column
                for var, var_index_in_x in vars_in_dir:
                    if var in data_temp.columns:
                        obs = data_temp[var].values
                    else:  # Fallback for special cases like precipitation_ana_gauges
                        obs = data_temp.iloc[:, 3].values

                    # Convert to float to handle NaN values properly
                    obs = obs.astype(float)
                    obs[obs < 0] = np.nan
                    x[k, target_indices, var_index_in_x] = obs[file_indices]
        return x

    def _read_ts_dynamic(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs,
    ):
        """Helper function to dynamically read time series data without caching."""
        if var_lst is None:
            return None
        # read_relevant_cols is now the unified reader for any time-series variables
        all_ts_data = self.read_relevant_cols(gage_id_lst, t_range, var_lst, **kwargs)

        times = pd.date_range(start=t_range[0], end=t_range[1], freq="D").values
        data_vars = {}
        for i, var in enumerate(var_lst):
            data_vars[var] = (("basin", "time"), all_ts_data[:, :, i])

        ds = xr.Dataset(data_vars, coords={"basin": gage_id_lst, "time": times})
        return ds

    def cache_timeseries_xrdataset(self, **kwargs):
        """Read time series data from cache or generate it and return an xarray.Dataset
        TODO: For p_ana_gauges, they are rainfall gauges, we need to calculate basin-averaged precipitation from them,
        if we want to use them as basin-averaged precipitation.

        """
        print("Creating cache for CAMELS-BR time series data... This may take a while.")
        all_basins = self.read_object_ids()
        all_vars = self._dynamic_features()
        # Define a canonical time range for the cache, e.g., 1980-2020
        canonical_t_range = self.default_t_range
        ds_full = self._read_ts_dynamic(
            gage_id_lst=all_basins,
            t_range=canonical_t_range,
            var_lst=all_vars,
            **kwargs,
        )
        ds_full.to_netcdf(self.cache_dir.joinpath(self._timeseries_cache_filename))
