import logging
import os
import collections
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
from hydrodataset import CAMELS_REGIONS
from hydrodataset.camels import Camels
from pandas.api.types import is_string_dtype, is_numeric_dtype

CAMELS_NO_DATASET_ERROR_LOG = (
    "We cannot read this dataset now. Please check if you choose correctly:\n"
    + str(CAMELS_REGIONS)
)


class CamelsBr(Camels):
    def __init__(
        self,
        data_path,
        download=False,
        region: str = "BR",
        version: str = "1.2",
        cache_path=None,
    ):
        """
        Initialization for CAMELS-BR dataset

        Parameters
        ----------
        data_path
            where we put the dataset.
            we already set the ROOT directory for hydrodataset,
            so here just set it as a relative path,
            by default "camels/camels_br"
        download
            if true, download, by default False
        region
            the default is CAMELS-BR
        cache_path
            the path to cache the dataset
        """
        self.data_path = os.path.join(data_path, "CAMELS_BR")
        super().__init__(self.data_path, download, region, cache_path=cache_path)
        # Build a map from variable name to its source directory
        self._variable_map = self._build_variable_map()

    @property
    def _attributes_cache_filename(self):
        return "camelsbr_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camelsbr_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1980-01-01", "2024-07-31"]

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
        camels_db = self.data_source_dir

        return self._set_data_source_camelsbr_describe(camels_db)

    def _set_data_source_camelsbr_describe(self, camels_db):
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

    def read_site_info(self) -> pd.DataFrame:
        """
        Read the basic information of gages in a CAMELS-BR dataset

        Returns
        -------
        pd.DataFrame
            basic info of gages
        """
        camels_file = self.data_source_description["CAMELS_GAUGE_FILE"]
        return pd.read_csv(camels_file, sep="\s+", dtype={"gauge_id": str})

    def get_constant_cols(self) -> np.ndarray:
        """
        all readable attrs in CAMELS-BR

        Returns
        -------
        np.array
            attribute types
        """
        data_folder = self.data_source_description["CAMELS_ATTR_DIR"]
        return self._get_constant_cols_some(data_folder, "camels_br_", ".txt", "\s+")

    def static_features(self):
        "Read static features list"
        return self.get_constant_cols()

    def dynamic_features(self):
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

    def read_attr_all(self):
        data_folder = self.data_source_description["CAMELS_ATTR_DIR"]
        key_lst = self.data_source_description["CAMELS_ATTR_KEY_LST"]
        f_dict = {}
        var_dict = {}
        var_lst = []
        out_lst = []
        gage_dict = self.sites
        camels_str = "camels_br_"
        sep_ = "\s+"
        for key in key_lst:
            data_file = os.path.join(data_folder, camels_str + key + ".txt")
            data_temp = pd.read_csv(data_file, sep=sep_)
            var_lst_temp = list(data_temp.columns[1:])
            var_dict[key] = var_lst_temp
            var_lst.extend(var_lst_temp)
            k = 0
            gage_id_key = "gauge_id"
            n_gage = len(gage_dict[gage_id_key].values)
            out_temp = np.full([n_gage, len(var_lst_temp)], np.nan)
            for field in var_lst_temp:
                if is_string_dtype(data_temp[field]):
                    value, ref = pd.factorize(data_temp[field], sort=True)
                    out_temp[:, k] = value
                    f_dict[field] = ref.tolist()
                elif is_numeric_dtype(data_temp[field]):
                    out_temp[:, k] = data_temp[field].values
                k = k + 1
            out_lst.append(out_temp)
        out = np.concatenate(out_lst, 1)
        return out, var_lst, var_dict, f_dict

    def read_constant_cols(
        self,
        gage_id_lst=None,
        var_lst=None,
        is_return_dict=False,
        **kwargs,
    ) -> np.ndarray:
        """
        Read Attributes data

        Parameters
        ----------
        gage_id_lst
            station ids
        var_lst
            attribute variable types
        is_return_dict
            if true, return var_dict and f_dict for CAMELS_US
        Returns
        -------
        Union[tuple, np.array]
            if attr var type is str, return factorized data.
            When we need to know what a factorized value represents, we need return a tuple;
            otherwise just return an array
        """
        attr_all, var_lst_all, var_dict, f_dict = self.read_attr_all()
        ind_var = [var_lst_all.index(var) for var in var_lst]
        id_lst_all = self.read_object_ids()
        # Notice the sequence of station ids ! Some id_lst_all are not sorted, so don't use np.intersect1d
        ind_grid = [id_lst_all.tolist().index(tmp) for tmp in gage_id_lst]
        temp = attr_all[ind_grid, :]
        out = temp[:, ind_var]
        return (out, var_dict, f_dict) if is_return_dict else out

    def read_area(self, gage_id_lst) -> np.ndarray:
        return self.read_constant_cols(gage_id_lst, ["area"], is_return_dict=False)

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

    def _read_attr_static(self, gage_id_lst=None, var_lst=None, **kwargs):
        """Helper function to dynamically read attribute data without caching."""
        if var_lst is None or len(var_lst) == 0:
            return None

        attr_data, var_dict, f_dict = self.read_constant_cols(
            gage_id_lst=gage_id_lst, var_lst=var_lst, is_return_dict=True
        )

        data_vars = {}
        for i, var in enumerate(var_lst):
            da = xr.DataArray(
                attr_data[:, i], dims=["basin"], coords={"basin": gage_id_lst}
            )
            if var in f_dict:
                da.attrs["category_mapping"] = str(f_dict[var])
            data_vars[var] = da

        ds = xr.Dataset(data_vars)
        return ds

    def cache_timeseries_xrdataset(self, **kwargs):
        """Read time series data from cache or generate it and return an xarray.Dataset
        TODO: For p_ana_gauges, they are rainfall gauges, we need to calculate basin-averaged precipitation from them,
        if we want to use them as basin-averaged precipitation.

        """
        print("Creating cache for CAMELS-BR time series data... This may take a while.")
        all_basins = self.read_object_ids()
        all_vars = self.dynamic_features()
        # Define a canonical time range for the cache, e.g., 1980-2020
        canonical_t_range = self.default_t_range
        ds_full = self._read_ts_dynamic(
            gage_id_lst=all_basins,
            t_range=canonical_t_range,
            var_lst=all_vars,
            **kwargs,
        )
        ds_full.to_netcdf(self.cache_dir.joinpath(self._timeseries_cache_filename))

    def cache_attributes_xrdataset(self, **kwargs):
        """Read attribute data from cache or generate it and return an xarray.Dataset"""
        print("Creating cache for CAMELS-BR attributes data...")
        all_basins = self.read_object_ids()
        all_vars = self.get_constant_cols()
        ds_full = self._read_attr_static(
            gage_id_lst=all_basins, var_lst=all_vars, **kwargs
        )
        ds_full.to_netcdf(self.cache_dir.joinpath(self._attributes_cache_filename))

    def read_mean_prcp(self, gage_id_lst, unit="mm/d") -> xr.Dataset:
        """Read mean precipitation data

        Parameters
        ----------
        gage_id_lst : list
            station ids
        unit : str, optional
            the unit of mean_prcp, by default "mm/d"

        Returns
        -------
        xr.Dataset
            mean precipitation data
        """
        data = self.read_constant_cols(
            gage_id_lst,
            ["p_mean"],
            is_return_dict=False,
        )
        if unit in ["mm/d", "mm/day"]:
            converted_data = data
        elif unit in ["mm/h", "mm/hour"]:
            converted_data = data / 24
        elif unit in ["mm/3h", "mm/3hour"]:
            converted_data = data / 8
        elif unit in ["mm/8d", "mm/8day"]:
            converted_data = data * 8
        else:
            raise ValueError(
                "unit must be one of ['mm/d', 'mm/day', 'mm/h', 'mm/hour', 'mm/3h', 'mm/3hour', 'mm/8d', 'mm/8day']"
            )
        return converted_data
