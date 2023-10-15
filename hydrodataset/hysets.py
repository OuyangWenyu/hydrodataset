import collections
import os
from typing import Union
import xarray as xr
import warnings
import pandas as pd
import numpy as np
from hydroutils import hydro_file
from hydrodataset import HydroDataset


class Hysets(HydroDataset):
    def __init__(self, data_path, download=False, region="NA"):
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
        """
        super().__init__(data_path)
        self.data_source_description = self.set_data_source_describe()
        if download:
            self.download_data_source()
        self.sites = self.read_site_info()

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

    def read_target_cols(
        self,
        gage_id_lst: Union[list, np.array] = None,
        t_range: list = None,
        target_cols: Union[list, np.array] = None,
        **kwargs
    ) -> np.array:
        return self._read_timeseries_data(
            "FLOW_FILE", gage_id_lst, t_range, target_cols
        )

    def read_relevant_cols(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        forcing_type="daymet",
    ) -> np.array:
        """_summary_

        Parameters
        ----------
        gage_id_lst : list, optional
            _description_, by default None
        t_range : list, optional
            A special notice is that for xarray, the time range is [start, end] which is a closed interval.
        var_lst : list, optional
            _description_, by default None
        forcing_type : str, optional
            _description_, by default "daymet"

        Returns
        -------
        np.array
            _description_
        """
        return self._read_timeseries_data("FORCING_FILE", gage_id_lst, t_range, var_lst)

    def _read_timeseries_data(self, file_name, gage_id_lst, t_range, var_lst):
        ts_file = self.data_source_description[file_name]
        data = xr.open_dataset(ts_file)
        gage_id_lst = [int(id) for id in gage_id_lst]
        if gage_id_lst is not None:
            watershed_ids = [float(id) for id in gage_id_lst]
            data = data.where(data.watershedID.isin(watershed_ids), drop=True)
        if t_range is not None:
            data = data.sel(time=slice(t_range[0], t_range[1]))
        if var_lst is None:
            if file_name == "FLOW_FILE":
                var_lst = self.get_target_cols()
            else:
                var_lst = self.get_relevant_cols()
        return data[var_lst]

    def read_constant_cols(
        self, gage_id_lst=None, var_lst=None, is_return_dict=False
    ) -> Union[tuple, np.array]:
        attr_file = self.data_source_description["ATTR_FILE"]
        data = pd.read_csv(attr_file, sep=",", dtype={"Watershed_ID": str})
        data = data.set_index("Watershed_ID")
        if gage_id_lst is not None:
            data = data.loc[gage_id_lst]
        if var_lst is not None:
            data = data.loc[:, var_lst]
        return data.to_dict("index") if is_return_dict else data.values
