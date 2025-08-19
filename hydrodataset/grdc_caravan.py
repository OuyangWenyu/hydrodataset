# """
# Author: Wenyu Ouyang
# Date: 2024-12-30 18:44:19
# LastEditTime: 2025-01-06 08:23:02
# LastEditors: Wenyu Ouyang
# Description: For GRDC-Caravan dataset
# FilePath: \hydrodataset\hydrodataset\grdc_caravan.py
# Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
# """

# import collections
# import os
# import tarfile
# import warnings
# import tqdm
# from hydrodataset.caravan import Caravan


# class GrdcCaravan(Caravan):
#     def __init__(self, data_path, download=False):
#         """
#         Initialization for GRDC-Caravan dataset

#         Parameters
#         ----------
#         data_path
#             where we put the dataset
#         download
#             if true, download
#         """
#         super().__init__(data_path, download=download, region="GRDC")

#     @property
#     def region_name_dict(self):
#         _region_name_dict = super().region_name_dict
#         _region_name_dict["GRDC"] = "grdc"
#         return _region_name_dict

#     def get_name(self):
#         return "GRDC-Caravan"

#     def set_data_source_describe(self) -> collections.OrderedDict:
#         """
#         Introduce the files in the dataset and list their location in the file system

#         Returns
#         -------
#         collections.OrderedDict
#             the description for GRDC-Caravan
#         """
#         the_dict = super().set_data_source_describe()
#         # Here we use nc files
#         the_dict["DOWNLOAD_URL"] = [
#             "https://zenodo.org/records/14006282/files/caravan-grdc-extension-csv.tar.gz",
#             "https://zenodo.org/records/14006282/files/caravan-grdc-extension-nc.tar.gz",
#             "https://zenodo.org/records/14006282/files/grdc-caravan_data_description.pdf",
#         ]
#         the_dict["TS_CSV_DIR"] = os.path.join(
#             self.data_source_dir,
#             "GRDC-Caravan-extension-csv",
#             "timeseries",
#             "csv",
#         )
#         return the_dict

#     def _base_dir(self):
#         # we use csv directory to read the data
#         return os.path.join(self.data_source_dir, "GRDC-Caravan-extension-nc")

#     def download_data_source(self) -> None:
#         """
#         Download dataset.

#         Returns
#         -------
#         None
#         """
#         self.data_source_dir.mkdir(exist_ok=True)
#         print(
#             "We only support manual downloading now. Please download two tar.gz files and one pdf file from the download links below:"
#             + "https://zenodo.org/records/14006282/files/caravan-grdc-extension-csv.tar.gz\n"
#             + "https://zenodo.org/records/14006282/files/caravan-grdc-extension-nc.tar.gz \n"
#             + "https://zenodo.org/records/14006282/files/grdc-caravan_data_description.pdf"
#         )
#         for zipfile in self.data_source_description["DOWNLOAD_URL"][:-1]:
#             file_name = os.path.join(self.data_source_dir, zipfile.split("/")[-1])
#             try:
#                 with tarfile.open(file_name, "r:gz") as tar:
#                     # Create a tqdm progress bar, assuming the total number of files equals the number of members in the tar archive (this may not be accurate).
#                     with tqdm.tqdm(
#                         total=len(tar.getmembers()), desc="Extracting"
#                     ) as pbar:
#                         for member in tar.getmembers():
#                             # We are not actually extracting each file to update the progress bar, because that would be too slow. Instead, we simulate progress updates (which does not reflect actual progress).
#                             tar.extract(
#                                 member, path=self.data_source_dir
#                             )  # extract the file
#                             pbar.update(1)  # update the progress bar
#             except tarfile.ReadError:
#                 warnings.warn("Please manually unzip the file.")
"""
Author: Wenyu Ouyang
Date: 2024-12-30 18:44:19
LastEditTime: 2025-01-06 08:23:02
LastEditors: Wenyu Ouyang
Description: For GRDC-Caravan dataset
FilePath: \hydrodataset\hydrodataset\grdc_caravan.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import logging
import glob
import collections
import os
import tarfile
import warnings
import tqdm
import pandas as pd
import numpy as np
import xarray as xr
from typing import Optional, Union
import pint_xarray  # noqa
import pint
from pint import UnitRegistry
from hydrodataset import CACHE_DIR
import re


class GrdcCaravan:
    def __init__(self, data_path, download=False):
        """
        Initialization for GRDC-Caravan dataset

        Parameters
        ----------
        data_path : str
            Path where the dataset is stored.
        download : bool
            If True, download the dataset.
        """
        self.data_source_dir = data_path
        self.download = download
        self.data_source_description = self.set_data_source_describe()
        self.site = None  # to store site information

        self.sites = self.read_site_info()

        if download:
            self.download_data_source()

    def get_name(self):
        """
        Get the name of the dataset.

        Returns
        -------
        str
            Name of the dataset.
        """
        return "GRDC-Caravan"

    def set_data_source_describe(self) -> collections.OrderedDict:
        """
        Introduce the files in the dataset and list their location in the file system.

        Returns
        -------
        collections.OrderedDict
            The description for GRDC-Caravan.
        """
        the_dict = collections.OrderedDict()
        the_dict["DOWNLOAD_URL"] = [
            "https://zenodo.org/records/15349031/files/GRDC_Caravan_extension_nc.zip?download=1",
            "https://zenodo.org/records/15349031/files/GRDC_Caravan_extension_csv.zip?download=1",
            "https://zenodo.org/records/15349031/files/grdc-caravan_data_description.pdf?download=1",
        ]
        the_dict["TS_CSV_DIR"] = os.path.join(
            self.data_source_dir,
            "GRDC-Caravan-extension-csv",
            "timeseries",
            "csv",
        )
        return the_dict

    def _base_dir(self):
        """
        Get the base directory for the dataset.

        Returns
        -------
        str
            Base directory path.
        """
        return os.path.join(self.data_source_dir, "GRDC-Caravan-extension-nc")

    def download_data_source(self) -> None:
        """
        Download the dataset.

        Returns
        -------
        None
        """
        os.makedirs(self.data_source_dir, exist_ok=True)
        print(
            "We only support manual downloading now. Please download two tar.gz files and one pdf file from the download links below:\n"
            + "https://zenodo.org/records/15349031/files/GRDC_Caravan_extension_nc.zip?download=1\n"
            + "https://zenodo.org/records/15349031/files/GRDC_Caravan_extension_csv.zip?download=1\n"
            + "https://zenodo.org/records/15349031/files/grdc-caravan_data_description.pdf?download=1"
        )
        for zipfile in self.data_source_description["DOWNLOAD_URL"][:-1]:
            file_name = os.path.join(self.data_source_dir, zipfile.split("/")[-1])
            try:
                with tarfile.open(file_name, "r:gz") as tar:
                    # Create a tqdm progress bar
                    with tqdm.tqdm(
                        total=len(tar.getmembers()), desc="Extracting"
                    ) as pbar:
                        for member in tar.getmembers():
                            tar.extract(member, path=self.data_source_dir)
                            pbar.update(1)
            except tarfile.ReadError:
                warnings.warn("Please manually unzip the file.")
            except Exception as e:
                warnings.warn(f"An error occurred: {e}")

    def read_site_info(self) -> pd.DataFrame:
        """
        Read the basic information of gauges in the dataset.

        Returns
        -------
        pd.DataFrame
            Basic information of gauges.
        """
        # path to the csv file containing site information
        site_file = os.path.join(
            self.data_source_dir,
            "attributes",
            "grdc",
            "attributes_other_grdc.csv",
        )
        # check if the files existts
        if not os.path.exists(site_file):
            raise FileNotFoundError(f"Site information file not found: {site_file}")
        # Read the site information CSV file into pandas DataFrame
        site_info = pd.read_csv(site_file, sep=",")
        return site_info

    def get_constant_cols(self) -> np.array:
        """
        Read constant columns (attributes) for GRDC-Caravan dataset.

        Returns
        -------
        np.array
            Constant columns for all gauges.
        """
        # path to grdc attributes files
        attr_file1 = os.path.join(
            self.data_source_dir, "attributes", "grdc", "attributes_caravan_grdc.csv"
        )
        attr_file2 = os.path.join(
            self.data_source_dir, "attributes", "grdc", "attributes_hydroatlas_grdc.csv"
        )
        attr_file3 = os.path.join(
            self.data_source_dir, "attributes", "grdc", "attributes_other_grdc.csv")
        # Check if the files exist
        if not os.path.exists(attr_file1):
            raise FileNotFoundError(f"Attribute file not found: {attr_file1}")
        if not os.path.exists(attr_file2):
            raise FileNotFoundError(f"Attribute file not found: {attr_file2}")
        if not os.path.exists(attr_file3):
            raise FileNotFoundError(f"Attribute file not found: {attr_file3}")
        attr_data1 = pd.read_csv(attr_file1, sep=",", dtype={"gauge_id": str})
        attr_data2 = pd.read_csv(attr_file2, sep=",", dtype={"gauge_id": str})
        attr_data3 = pd.read_csv(attr_file3, sep=",", dtype={"gauge_id": str})
        # concatenate the columns excluding gauge_id
        attr_columns = (
            attr_data1.columns.values[1:].tolist()
            + attr_data2.columns.values[1:].tolist()
            + attr_data3.columns.values[1:].tolist()
        )
        # return unique attributes types as numpy array
        return np.array(attr_columns)

    def get_relevant_cols(self) -> np.array:
        """
        Get relevant forcing columns for GRDC-Caravan dataset.

        Returns
        -------
        np.array
            Relevant forcing from the NETCDF files.
        """
        # path to grdc netcdf firectory
        forcing_dir = os.path.join(self.data_source_dir, "timeseries", "netcdf", "grdc")
        # check if directory exists
        if not os.path.exists(forcing_dir):
            raise FileNotFoundError(f"Forcing directory not found: {forcing_dir}")
        # list all netcdf files in the directory
        files = [f for f in os.listdir(forcing_dir) if f.endswith(".nc")]
        if not files:
            raise FileNotFoundError(
                f"No netcdf files found in directory: {forcing_dir}"
            )
        # open the first netcdf file to get the variable names
        first_file = os.path.join(forcing_dir, files[0])
        data = xr.open_dataset(first_file)

        return np.array(data.data_vars)

    def get_target_cols(self) -> np.array:
        """
        the target vars are streamflows

        Returns
        -------
        np.array
            streamflow types
        """
        return np.array(["streamflow"])

    def read_object_ids(self, **kwargs) -> np.array:
        """
        Read station IDs.

        Parameters
        ----------
        **kwargs
            Optional parameters if needed.

        Returns
        -------
        np.array
            Gage/station IDs.
        """
        if self.site is None:
            self.site = self.read_site_info()

        if self.site is None:
            self.site = self.read_site_info()

        if self.site is None:
            raise ValueError("Site metadata has not been loaded.")
        return self.site["gauge_id"].values

    def read_target_cols(
        self,
        gage_id_lst: Union[list, np.array] = None,
        t_range: list = None,
        target_cols: Union[list, np.array] = None,
        **kwargs,
    ) -> np.array:
        return self._read_timeseries_data(
            dir_name="timeseries/netcdf/grdc",
            gage_id_lst=gage_id_lst,
            t_range=t_range,
            var_lst=target_cols,
            default_vars="target",
        )

    def read_relevant_cols(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        forcing_type="grdc",
    ) -> np.array:
        return self._read_timeseries_data(
            dir_name="timeseries/netcdf/grdc",
            gage_id_lst=gage_id_lst,
            t_range=t_range,
            var_lst=var_lst,
            default_vars="relevant",
        )

    def _read_timeseries_data(
        self,
        gage_id_lst=None,
        t_range=None,
        var_lst=None,
        default_vars="relevant",
        dir_name=None,
    ):
        """
        Read time series data for the GRDC-Caravan dataset.

        Parameters
        ----------
        gage_id_lst : list[str], optional
            List of basin IDs (file names without .nc). If None, reads all available.
        t_range : list[str], optional
            Time range as [start_date, end_date].
        var_lst : list[str], optional
            Variables to read. If None, will fallback to default_vars.
        default_vars : str, optional
            Either "target" or "relevant" (forcings+target).

        Returns
        -------
        xr.Dataset
            Time series data for the specified basins, time range, and variables.
        """
        if dir_name is None:
            dir_name = "timeseries/netcdf/grdc"
        # Folder where NC files are stored
        ts_dir = os.path.join(self.data_source_dir, dir_name)

        # Collect all gage IDs if none are specified
        if gage_id_lst is None:
            gage_id_lst = self.read_object_ids()

        # Match file paths
        file_paths = []
        for gid in gage_id_lst:
            path = os.path.join(ts_dir, f"{gid}.nc")
            if os.path.isfile(path):
                file_paths.append(path)
            else:
                raise FileNotFoundError(f"Missing NetCDF file: {path}")

        # Open datasets, attach gauge_id
        datasets = [
            xr.open_dataset(path).assign_coords(gauge_id=gid)
            for path, gid in zip(file_paths, gage_id_lst)
        ]
        data = xr.concat(datasets, dim="gauge_id")

        # Apply date slicing
        if t_range is not None:
            data = data.sel(date=slice(t_range[0], t_range[1]))

        # Decide which vars to return
        if var_lst is None:
            if default_vars == "target":
                var_lst = self.get_target_cols()
            elif default_vars == "relevant":
                var_lst = self.get_relevant_cols()
            else:
                raise ValueError(f"Unknown default_vars='{default_vars}'")

        # Filter to requested variables
        return data[var_lst]

    def read_constant_cols(
        self, gage_id_lst=None, var_lst=None, is_return_dict=False
    ) -> Union[tuple, np.array]:
        """
        Read Attributes data for the GRDC-Caravan dataset.

        Parameters
        ----------
        gage_id_lst : list, optional
            List of station IDs to read attributes for. If None, reads all stations.
        var_lst : list, optional
            List of attribute variable types to read. If None, reads all attributes.
        is_return_dict : bool, optional
            If True, returns data as a dictionary. Otherwise, returns as a NumPy array.

        Returns
        -------
        Union[tuple, np.array]
            Attribute data as a dictionary or NumPy array.
        """
        data = self._read_attr_files_grdc(gage_id_lst, var_lst)
        return data.to_dict("index") if is_return_dict else data.values

    def _read_attr_files_grdc(self, gage_id_lst, var_lst):
        """
        Read attribute files for the GRDC-Caravan dataset, now including 'area'.

        Parameters
        ----------
        gage_id_lst : list, optional
            List of station IDs to read attributes for. If None, reads all stations.
        var_lst : list, optional
            List of attribute variable types to read. If None, reads all attributes.

        Returns
        -------
        pd.DataFrame
            Attributes data.
        """
        # Paths to the GRDC attribute files
        attr_file1 = os.path.join(
            self.data_source_dir,
            "attributes",
            "grdc",
            "attributes_caravan_grdc.csv",
        )
        attr_file2 = os.path.join(
            self.data_source_dir,
            "attributes",
            "grdc",
            "attributes_hydroatlas_grdc.csv",
        )
        attr_file3 = os.path.join(
            self.data_source_dir,
            "attributes",
            "grdc",
            "attributes_other_grdc.csv",   # <-- contains 'area'
        )

        # Read the attribute files
        data1 = pd.read_csv(attr_file1, sep=",", dtype={"gauge_id": str}).set_index("gauge_id")
        data2 = pd.read_csv(attr_file2, sep=",", dtype={"gauge_id": str}).set_index("gauge_id")
        data3 = pd.read_csv(attr_file3, sep=",", dtype={"gauge_id": str}).set_index("gauge_id")

        # Concatenate the data
        data = pd.concat([data1, data2, data3], axis=1)
 

        # Filter by gage IDs if provided
        if gage_id_lst is not None:
            data = data.loc[gage_id_lst]

        # Filter by variables if provided
        if var_lst is not None:
            data = data.loc[:, var_lst]

        return data
    
    def read_mean_prep(self, object_ids) -> np.array:
        return self.read_constant_cols(object_ids, ["p_mean"], is_return_dict=False)

    def cache_attributes_xrdataset(self, chunk_size=1000):
        """
        Cache attribute variables as an xarray.Dataset using Dask for parallelization.
        Skips caching if the file already exists.
        """
        import dask.array as da

        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)
        file_path = os.path.join(CACHE_DIR, "grdc_caravan_attributes.nc")  # <- define first

        # Skip caching if file already exists
        if os.path.isfile(file_path):
            print(f"Cached attributes already exist at {file_path}, skipping.")
            return xr.open_dataset(file_path)

        # Read raw attribute data
        data = self.read_constant_cols()  # shape: (num_basins, num_attrs)
        basin_ids = self.read_object_ids()
        var_names = sorted(self.get_constant_cols())

        # Convert to Dask array (chunked along basins)
        data_dask = da.from_array(data, chunks=(chunk_size, -1))

        # Build xarray.Dataset lazily with Dask
        ds = xr.Dataset(
            {var: (["basin"], data_dask[:, i]) for i, var in enumerate(var_names)},
            coords={"basin": basin_ids},
        )

        # Define units
        units_dict = {attribute: "" for attribute in var_names}
        units_dict["p_mean"] = "mm/year"
        units_dict["area"] = "km^2"
        units_dict["aridity"] = "dimensionless"

        from pint import UnitRegistry
        ureg = UnitRegistry()
        pint_unit_mapping = {
            "mm/year": "millimeter/year",
            "km^2": "kilometer ** 2",
            "dimensionless": "dimensionless",
        }

        converted_units = {var: pint_unit_mapping.get(unit, unit) for var, unit in units_dict.items()}

        for var_name in converted_units:
            if var_name in ds.data_vars:
                ds[var_name].attrs["units"] = converted_units[var_name]

        # Save the dataset to cache
        ds.to_netcdf(file_path, engine="netcdf4")
        print(f"Cached attributes saved to {file_path}")

        return ds



    def cache_timeseries_xrdataset(self, batch_size=1000):
        """
        Cache time series data in batches using Dask.
        Each batch writes a separate NetCDF file.
        """
        import warnings
        from dask.diagnostics import ProgressBar

        os.makedirs(CACHE_DIR, exist_ok=True)
        warnings.warn("Check the units of all variables.")

        ts_dir = os.path.join(self.data_source_dir, "timeseries", "netcdf", "grdc")
        gage_id_lst = self.read_object_ids()

        file_paths = [
            os.path.join(ts_dir, f"{file_name}.nc")
            for file_name in gage_id_lst
            if os.path.isfile(os.path.join(ts_dir, f"{file_name}.nc"))
        ]

        if not file_paths:
            raise FileNotFoundError(f"No NetCDF files found in {ts_dir}.")

        # Compute number of batches using simple integer division logic
        num_files = len(file_paths)
        num_batches = num_files // batch_size
        if num_files % batch_size > 0:
            num_batches += 1

        for i in range(num_batches):
            batch_files = file_paths[i * batch_size : (i + 1) * batch_size]
            batch_ids = gage_id_lst[i * batch_size : (i + 1) * batch_size]

            cache_ts_file = os.path.join(
                CACHE_DIR, f"grdc_caravan_timeseries_part_{i+1}.nc"
            )

            if os.path.isfile(cache_ts_file):
                print(f"Skipping existing {cache_ts_file}")
                continue

            combined_ds = xr.open_mfdataset(
                batch_files,
                combine="nested",
                concat_dim="gauge_id",
                parallel=True,
                chunks={"time": 365},
                engine="netcdf4",
            )

            combined_ds = combined_ds.assign_coords(gauge_id=batch_ids)

            with ProgressBar():
                combined_ds.to_netcdf(cache_ts_file, mode="w", format="NETCDF4")

            print(f"Saved batch {i+1}/{num_batches} → {cache_ts_file}")


    def read_attr_xrdataset(self, gage_id_lst=None, var_lst=None, **kwargs):
        file_path = os.path.join(CACHE_DIR, "grdc_caravan_attributes.nc")
        ds = xr.open_dataset(file_path)

        # Default to all basins if none provided
        if gage_id_lst is None:
            gage_id_lst = ds.basin.values

        ds = ds.sel(basin=gage_id_lst)

        if var_lst:
            ds = ds[var_lst]

        return ds


    def read_ts_xrdataset(self, gage_id_lst, t_range, var_lst, **kwargs):
        """
        Read Caravan time series NetCDF files into an xarray Dataset.
        """

        # 🔹 Step 1: Collect Caravan NetCDF files
        file_paths = sorted(
            glob.glob(os.path.join(CACHE_DIR, "*caravan*timeseries*.nc"))
        )
        assert file_paths, "No time series files found in the cache directory."

        # 🔹 Step 2: Open with dask for efficiency
        parallel = kwargs.get("parallel", False)
        combined_ds = xr.open_mfdataset(
            file_paths,
            combine="nested",
            concat_dim="gauge_id",
            parallel=parallel,
            chunks={"time": 365},
            engine="netcdf4"
        )

        # 🔹 Step 3: Check duplicate gauge IDs
        gauge_ids = pd.Series(combined_ds.gauge_id.values)
        duplicates = gauge_ids[gauge_ids.duplicated()]
        if not duplicates.empty:
            logging.error("Duplicate gauge IDs found: %s", duplicates.values)
            raise ValueError("Duplicate gauge IDs found. Please check your data extraction process.")
        else:
            logging.info("No duplicate gauge IDs found.")

        # 🔹 Step 4: Filter variables, time, and gauge IDs
        if var_lst:
            combined_ds = combined_ds[var_lst]
        if t_range:
            combined_ds = combined_ds.sel(date=slice(*t_range))
        if gage_id_lst:
            combined_ds = combined_ds.sel(gauge_id=gage_id_lst)

        # 🔹 Step 5: Handle missing units
        unit_mapping = {"W/m2": "watt / meter ** 2", "m3/m3": "meter^3/meter^3"}

        def extract_unit(variable_name, units_string):
            """Extract unit for a variable from the dataset metadata."""
            if variable_name == "streamflow":
                main_name = "streamflow"
            elif variable_name.startswith("dewpoint_temperature_2m"):
                main_name = "temperature_2m"
            else:
                main_name = "_".join(variable_name.split("_")[:-1])
            pattern = main_name + r":.*?\[(.*?)\]"
            match = re.search(pattern, units_string)
            return match[1].strip() if match else "unknown"

        for var in combined_ds.data_vars:
            if "units" not in combined_ds[var].attrs and "Units" in combined_ds.attrs:
                unit = extract_unit(var, combined_ds.attrs["Units"])
                unit = unit_mapping.get(unit, unit)
                combined_ds[var].attrs["units"] = unit

        # 🔹 Step 6: Rename dimensions consistently
        combined_ds = combined_ds.rename({"date": "time", "gauge_id": "basin"})

        return combined_ds


    @property
    def streamflow_unit(self):
        return "mm/d"

    def read_area(self, gage_id_lst=None):
        return self.read_attr_xrdataset(gage_id_lst, ["area"])

    def read_mean_prcp(self, gage_id_lst=None, unit="mm/d"):
        return self.read_attr_xrdataset(gage_id_lst, ["p_mean"])

    def check_coordinates(ds):
        """
        Ensure the dataset has 'basin' and 'time' as coordinates.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset to check.

        Raises
        ------
        ValueError
            If required coordinates are missing.
        """
        required_coords = ["basin", "time"]
        for coord in required_coords:
            if coord not in ds.coords:
                raise ValueError(f"Missing coordinate: {coord}")

    def _extract_unit(variable_name, units_string):
        """
        Extract the unit for a variable based on its name and the units string.

        Parameters
        ----------
        variable_name : str
            Name of the variable.
        units_string : str
            String containing unit information.

        Returns
        -------
        str
            Extracted unit or "unknown" if not found.
        """
        # Handle specific cases for variable names
        if variable_name == "streamflow":
            main_name = "streamflow"
        elif variable_name.startswith(
            "dewpoint_temperature_2m"
        ) or variable_name.startswith("temperature_2m"):
            main_name = "temperature_2m"
        else:
            main_name = "_".join(variable_name.split("_")[:-1])

        # Construct the regex pattern to extract the unit
        pattern = main_name + r":.*?\[(.*?)\]"
        match = re.search(pattern, units_string)
        the_unit = match[1].strip() if match else "unknown"

        # Manually set the unit for specific cases
        if the_unit == "unknown" and main_name == "streamflow":
            the_unit = "mm"  # Default unit for streamflow in GRDC-Caravan

        return the_unit


