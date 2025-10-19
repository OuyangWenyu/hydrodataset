import collections
import os
import xarray as xr
import warnings
import pandas as pd
import numpy as np
from hydroutils import hydro_file
from hydrodataset import HydroDataset


class Hysets(HydroDataset):
    def __init__(self, data_path, download=False, region="NA", cache_path=None):
        """
        Initialization for LamaH-CE dataset

        Parameters
        ----------
        data_path
            where we put the dataset
        download
            if true, download
        region
            the region is North America, NA
        cache_path
            the path to cache the dataset
        """
        super().__init__(data_path, cache_path=cache_path)
        self.data_source_description = self.set_data_source_describe()
        if download:
            self.download_data_source()
        self.sites = self.read_site_info()

    @property
    def _attributes_cache_filename(self):
        return "hysets_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "hysets_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1950-01-01", "2018-12-31"]

    def get_name(self):
        return "HYSETS"

    def set_data_source_describe(self) -> collections.OrderedDict:
        dataset_dir = self.data_source_dir
        shp_file = os.path.join(
            dataset_dir,
            "HYSETS_watershed_boundaries.zip",
        )
        # config of flow data
        flow_file = os.path.join(
            dataset_dir,
            "HYSETS_2020_QC_stations.nc",
        )
        forcing_file = flow_file
        attr_file = os.path.join(
            dataset_dir,
            "HYSETS_watershed_properties.txt",
        )

        return collections.OrderedDict(
            DATASET_DIR=dataset_dir,
            FLOW_FILE=flow_file,
            FORCING_FILE=forcing_file,
            ATTR_FILE=attr_file,
            BASINS_SHP_FILE=shp_file,
        )

    def download_data_source(self) -> None:
        """
        Download dataset.

        Returns
        -------
        None
        """
        warnings.warn(
            "We don't provide downloading methods for HYSETS yet. Please download all files manually from https://osf.io/rpc3w/"
        )
        hydro_file.zip_extract(self.data_source_description["DATASET_DIR"])

    def read_site_info(self) -> pd.DataFrame:
        attr_file = self.data_source_description["ATTR_FILE"]
        return pd.read_csv(attr_file, sep=",", dtype={"Watershed_ID": str})

    def get_constant_cols(self) -> np.array:
        attr_file = self.data_source_description["ATTR_FILE"]
        data = pd.read_csv(attr_file, sep=",", index_col=0)
        # Source, Name and Official_ID are not necessary for computing
        return data.columns.values.tolist()[3:]

    def get_relevant_cols(self) -> np.array:
        # now only for QCStations.nc file
        return ["pr", "tasmax", "tasmin"]

    def get_target_cols(self) -> np.array:
        return ["discharge"]

    def read_object_ids(self, **kwargs) -> np.array:
        return self.sites["Watershed_ID"].values

    def cache_attributes_xrdataset(self):
        attr_file = self.data_source_description["ATTR_FILE"]
        data = pd.read_csv(attr_file, sep=",", dtype={"Watershed_ID": str})
        data = data.set_index("Watershed_ID")
        ds = xr.Dataset.from_dataframe(data)
        ds.to_netcdf(self.cache_dir.joinpath(self._attributes_cache_filename))

    def cache_timeseries_xrdataset(self):
        ts_file = self.data_source_description["FLOW_FILE"]
        data = xr.open_dataset(ts_file)
        data.to_netcdf(self.cache_dir.joinpath(self._timeseries_cache_filename))