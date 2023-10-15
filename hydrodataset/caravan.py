import collections
import xarray as xr
import os
from pathlib import Path
from typing import Union
import tarfile
from urllib.request import urlopen
import pandas as pd
import numpy as np
from hydroutils import hydro_file
from hydrodataset import HydroDataset


class Caravan(HydroDataset):
    def __init__(self, data_path, download=False, region="US"):
        """
        Initialization for LamaH-CE dataset

        Parameters
        ----------
        data_path
            where we put the dataset
        download
            if true, download
        region
            the region can be US, AUS, BR, CL, GB, CE, NA (North America, same as HYSETS)
        """
        super().__init__(data_path)
        self.data_source_description = self.set_data_source_describe()
        self.region = region
        region_name_dict = {
            "US": "camels",
            "AUS": "camelsaus",
            "BR": "camelsbr",
            "CL": "camelscl",
            "GB": "camelsgb",
            "NA": "hysets",
            "CE": "lamah",
        }
        self.region_data_name = region_name_dict[region]
        if download:
            self.download_data_source()
        self.sites = self.read_site_info()

    def get_name(self):
        return "Caravan_" + self.region

    def set_data_source_describe(self) -> collections.OrderedDict:
        """
        Introduce the files in the dataset and list their location in the file system

        Returns
        -------
        collections.OrderedDict
            the description for LamaH-CE
        """
        dataset_dir = os.path.join(self.data_source_dir, "Caravan", "Caravan")

        # We use A_basins_total_upstrm
        # shp file of basins
        camels_shp_file = os.path.join(
            dataset_dir,
            "shapefiles",
        )
        # config of flow data
        flow_dir = os.path.join(dataset_dir, "timeseries", "netcdf")
        forcing_dir = flow_dir
        attr_dir = os.path.join(dataset_dir, "attributes")
        download_url = "https://zenodo.org/record/7944025/files/Caravan.zip"
        return collections.OrderedDict(
            DATASET_DIR=dataset_dir,
            FLOW_DIR=flow_dir,
            FORCING_DIR=forcing_dir,
            ATTR_DIR=attr_dir,
            BASINS_SHP_FILE=camels_shp_file,
            DOWNLOAD_URL=download_url,
        )

    def download_data_source(self) -> None:
        """
        Download dataset.

        Returns
        -------
        None
        """
        dataset_config = self.data_source_description
        self.data_source_dir.mkdir(exist_ok=True)
        url = dataset_config["DOWNLOAD_URL"]
        fzip = Path(self.data_source_dir, url.rsplit("/", 1)[1])
        if fzip.exists():
            with urlopen(url) as response:
                if int(response.info()["Content-length"]) != fzip.stat().st_size:
                    fzip.unlink()
        to_dl = []
        if not Path(self.data_source_dir, url.rsplit("/", 1)[1]).exists():
            to_dl.append(url)
        hydro_file.download_zip_files(to_dl, self.data_source_dir)
        # It seems that there is sth. wrong with hysets_06444000.nc
        try:
            hydro_file.zip_extract(dataset_config["DATASET_DIR"])
        except tarfile.ReadError:
            Warning("Please manually unzip the file.")

    def read_site_info(self) -> pd.DataFrame:
        """
        Read the basic information of gages in dataset

        Returns
        -------
        pd.DataFrame
            basic info of gages
        """
        site_file = os.path.join(
            self.data_source_description["ATTR_DIR"],
            self.region_data_name,
            "attributes_caravan_" + self.region_data_name + ".csv",
        )
        return pd.read_csv(site_file, sep=",")

    def get_constant_cols(self) -> np.array:
        """
        all readable attrs

        Returns
        -------
        np.array
            attribute types
        """
        attr_file = os.path.join(
            self.data_source_description["DATASET_DIR"],
            "attributes",
            self.region_data_name,
            "attributes_caravan_" + self.region_data_name + ".csv",
        )
        attr_indices_data = pd.read_csv(attr_file, sep=",")
        return attr_indices_data.columns.values[1:]

    def get_relevant_cols(self) -> np.array:
        """
        all readable forcing types

        Returns
        -------
        np.array
            forcing types
        """
        forcing_dir = os.path.join(
            self.data_source_description["FORCING_DIR"],
            self.region_data_name,
        )
        if not (files := os.listdir(forcing_dir)):
            raise FileNotFoundError("No files found in the directory.")
        first_file = files[0]
        file_path = os.path.join(forcing_dir, first_file)

        if files := os.listdir(forcing_dir):
            first_file = files[0]
            file_path = os.path.join(forcing_dir, first_file)
            data = xr.open_dataset(file_path)
        else:
            raise FileNotFoundError("No files found in the directory.")
        return list(data.data_vars)

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
        read station ids

        Parameters
        ----------
        **kwargs
            optional params if needed

        Returns
        -------
        np.array
            gage/station ids
        """
        return self.sites["gauge_id"].values

    def read_target_cols(
        self,
        gage_id_lst: Union[list, np.array] = None,
        t_range: list = None,
        target_cols: Union[list, np.array] = None,
        **kwargs,
    ) -> np.array:
        return self._read_timeseries_data("FLOW_DIR", gage_id_lst, t_range, target_cols)

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
        return self._read_timeseries_data("FORCING_DIR", gage_id_lst, t_range, var_lst)

    def _read_timeseries_data(self, dir_name, gage_id_lst, t_range, var_lst):
        ts_dir = self.data_source_description[dir_name]
        if gage_id_lst is None:
            gage_id_lst = self.read_object_ids()
        # Find matching file paths
        file_paths = []
        for file_name in gage_id_lst:
            file_path = os.path.join(ts_dir, self.region_data_name, file_name) + ".nc"
            if os.path.isfile(file_path):
                file_paths.append(file_path)
        datasets = [
            xr.open_dataset(path).assign_coords(gauge_id=name)
            for path, name in zip(file_paths, gage_id_lst)
        ]
        # Concatenate the datasets along the new dimension
        data = xr.concat(datasets, dim="gauge_id")
        if t_range is not None:
            data = data.sel(date=slice(t_range[0], t_range[1]))
        if var_lst is None:
            if dir_name == "FLOW_DIR":
                var_lst = self.get_target_cols()
            else:
                var_lst = self.get_relevant_cols()
        return data[var_lst]

    def read_constant_cols(
        self, gage_id_lst=None, var_lst=None, is_return_dict=False
    ) -> Union[tuple, np.array]:
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
        attr_file = os.path.join(
            self.data_source_description["DATASET_DIR"],
            "attributes",
            self.region_data_name,
            "attributes_caravan_" + self.region_data_name + ".csv",
        )
        data = pd.read_csv(attr_file, sep=",", dtype={"gauge_id": str})
        data = data.set_index("gauge_id")
        if gage_id_lst is not None:
            data = data.loc[gage_id_lst]
        if var_lst is not None:
            data = data.loc[:, var_lst]
        return data.to_dict("index") if is_return_dict else data.values

    def read_basin_area(self, object_ids) -> np.array:
        return self.read_constant_cols(object_ids, ["area_calc"], is_return_dict=False)

    def read_mean_prep(self, object_ids) -> np.array:
        return self.read_constant_cols(object_ids, ["p_mean"], is_return_dict=False)
