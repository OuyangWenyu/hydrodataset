"""
Author: Wenyu Ouyang
Date: 2022-01-05 18:01:11
LastEditTime: 2023-10-15 14:51:13
LastEditors: Wenyu Ouyang
Description: Read Camels datasets
FilePath: \hydrodataset\hydrodataset\mopex.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import collections
import fnmatch
import os
import pandas as pd
import numpy as np
import xarray as xr
from pandas.core.dtypes.common import is_string_dtype, is_numeric_dtype
from tqdm import tqdm
from hydroutils import hydro_file, hydro_time
from hydrodataset.hydro_dataset import HydroDataset


class Mopex(HydroDataset):
    def __init__(self, data_path, download=False, cache_path=None):
        """
        Initialization for dataset

        Parameters
        ----------
        data_path
            where we put the dataset
        download
            if true, download
        cache_path
            the path to cache the dataset
        """
        super().__init__(data_path, cache_path=cache_path)
        self.data_source_description = self.set_data_source_describe()
        if download:
            self.download_data_source()
        self.camels_sites = self.read_site_info()

    @property
    def _attributes_cache_filename(self):
        return "mopex_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "mopex_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1950-01-01", "2013-12-31"]

    def get_name(self):
        return "CANOPEX"

    def set_data_source_describe(self) -> collections.OrderedDict:
        """
        Introduce the files in the dataset and list their location in the file system

        Returns
        -------
        collections.OrderedDict
            the description for dataset
        """
        camels_db = self.data_source_dir
        # shp file of basins
        camels_shp_files_dir = os.path.join(camels_db, "CANOPEX_BOUNDARIES")
        # config of flow data
        flow_dir = os.path.join(camels_db, "CANOPEX_NRCAN_ASCII", "CANOPEX_NRCAN_ASCII")
        forcing_dir = flow_dir
        # There is no attr data in CANOPEX, hence we use attr from HYSET -- https://osf.io/7fn4c/
        attr_dir = camels_db

        gauge_id_file = os.path.join(camels_db, "STATION_METADATA.xlsx")

        return collections.OrderedDict(
            CAMELS_DIR=camels_db,
            CAMELS_FLOW_DIR=flow_dir,
            CAMELS_FORCING_DIR=forcing_dir,
            CAMELS_ATTR_DIR=attr_dir,
            CAMELS_GAUGE_FILE=gauge_id_file,
            CAMELS_BASINS_SHP_DIR=camels_shp_files_dir,
        )

    def download_data_source(self) -> None:
        """
        Download dataset

        Returns
        -------
        None
        """
        camels_config = self.data_source_description
        for f_name in os.listdir(camels_config["CAMELS_DIR"]):
            if fnmatch.fnmatch(f_name, "*.zip"):
                unzip_dir = os.path.join(camels_config["CAMELS_DIR"], f_name[:-4])
                file_name = os.path.join(camels_config["CAMELS_DIR"], f_name)
                hydro_file.unzip_nested_zip(file_name, unzip_dir)

    def read_site_info(self) -> pd.DataFrame:
        """
        Read the basic information of gages in dataset

        Returns
        -------
        pd.DataFrame
            basic info of gages
        """
        camels_file = self.data_source_description["CAMELS_GAUGE_FILE"]
        return pd.read_excel(camels_file)

    def get_constant_cols(self) -> np.array:
        """
        all readable attrs

        Returns
        -------
        np.array
            attribute types
        """
        attr_all_file = os.path.join(
            self.data_source_description["CAMELS_DIR"],
            "HYSETS_watershed_properties.txt",
        )
        canopex_attr_indices_data = pd.read_csv(attr_all_file, sep=";")
        # exclude HYSETS watershed id
        return canopex_attr_indices_data.columns.values[3:]

    def get_relevant_cols(self) -> np.array:
        """
        all readable forcing types

        Returns
        -------
        np.array
            forcing types
        """
        # Although there is climatic potential evaporation item, CANOPEX does not have any PET data
        return np.array(["prcp", "tmax", "tmin"])

    def get_target_cols(self) -> np.array:
        """
        the target vars are streamflows

        Returns
        -------
        np.array
            streamflow types
        """
        return np.array(["discharge"])

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
        ids = self.camels_sites["STATION_ID"].values
        id_strs = [id_.split("'")[1] for id_ in ids]
        # although there are 698 sites, there are only 611 sites with attributes data.
        # Hence we only use 611 sites now
        attr_all_file = os.path.join(
            self.data_source_description["CAMELS_DIR"],
            "HYSETS_watershed_properties.txt",
        )
        if not os.path.isfile(attr_all_file):
            raise FileNotFoundError(
                "Please download HYSETS_watershed_properties.txt from https://osf.io/7fn4c/ and put it in the "
                "root directory of CANOPEX"
            )
        canopex_attr_data = pd.read_csv(attr_all_file, sep=";")
        return np.intersect1d(id_strs, canopex_attr_data["Official_ID"].values)

    def cache_attributes_xrdataset(self):
        attr_all_file = os.path.join(
            self.data_source_description["CAMELS_ATTR_DIR"],
            "HYSETS_watershed_properties.txt",
        )
        all_attr_tmp = pd.read_csv(attr_all_file, sep=";", index_col=0)
        all_attr = all_attr_tmp[
            all_attr_tmp["Official_ID"].isin(self.read_object_ids())
        ]
        var_lst = self.get_constant_cols().tolist()
        data_temp = all_attr[var_lst]
        n_gage = len(self.read_object_ids())
        out = np.full([n_gage, len(var_lst)], np.nan)
        f_dict = {}
        k = 0
        for field in var_lst:
            if is_string_dtype(data_temp[field]):
                value, ref = pd.factorize(data_temp[field], sort=True)
                out[:, k] = value
                f_dict[field] = ref.tolist()
            elif is_numeric_dtype(data_temp[field]):
                out[:, k] = data_temp[field].values
            k = k + 1
        ds = xr.Dataset(
            {var: (["basin"], out[:, i]) for i, var in enumerate(var_lst)},
            coords={"basin": self.read_object_ids()},
        )
        for field, ref in f_dict.items():
            ds[field].attrs["category_mapping"] = str(ref)
        ds.to_netcdf(self.cache_dir.joinpath(self._attributes_cache_filename))

    def cache_timeseries_xrdataset(self):
        t_range_list = hydro_time.t_range_days(self.default_t_range)
        nt = t_range_list.shape[0]
        var_lst = self.get_relevant_cols().tolist() + self.get_target_cols().tolist()
        y = np.full([len(self.read_object_ids()), nt, len(var_lst)], np.nan)

        for k, gage_id in enumerate(
            tqdm(self.read_object_ids(), desc="Read streamflow data of CANOPEX")
        ):
            canopex_id = self.camels_sites[
                self.camels_sites["STATION_ID"] == "'" + gage_id + "'"
            ]["CANOPEX_ID"].values[0]
            flow_file = os.path.join(
                self.data_source_description["CAMELS_FLOW_DIR"],
                f"{str(canopex_id)}.dly",
            )
            read_flow_file = pd.read_csv(flow_file, header=None).values.tolist()
            flow_data = []
            flow_date = []
            prcp_data = []
            tmax_data = []
            tmin_data = []
            for one_site in read_flow_file:
                flow_date.append(
                    hydro_time.t2dt(int(one_site[0][:8].replace(" ", "0")))
                )
                all_data = one_site[0].split(" ")
                real_data = [one_data for one_data in all_data if one_data != ""]
                flow_data.append(float(real_data[-3]))
                prcp_data.append(float(real_data[-5]))
                tmax_data.append(float(real_data[-2]))
                tmin_data.append(float(real_data[-1]))
            date = pd.to_datetime(flow_date).values.astype("datetime64[D]")
            [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
            flow_obs = np.array(flow_data)
            flow_obs[flow_obs < 0] = np.nan
            y[k, ind2, var_lst.index("discharge")] = flow_obs[ind1] * 35.314666721489
            prcp_obs = np.array(prcp_data)
            y[k, ind2, var_lst.index("prcp")] = prcp_obs[ind1]
            tmax_obs = np.array(tmax_data)
            y[k, ind2, var_lst.index("tmax")] = tmax_obs[ind1]
            tmin_obs = np.array(tmin_data)
            y[k, ind2, var_lst.index("tmin")] = tmin_obs[ind1]

        ds = xr.Dataset(
            {var: (["basin", "time"], y[:, :, i]) for i, var in enumerate(var_lst)},
            coords={"basin": self.read_object_ids(), "time": t_range_list},
        )
        ds.to_netcdf(self.cache_dir.joinpath(self._timeseries_cache_filename))

    def read_area(self, object_ids) -> np.array:
        return self.read_constant_cols(
            object_ids, ["Drainage_Area_km2"], is_return_dict=False
        )

    def read_mean_prep(self, object_ids) -> np.array:
        # There is no p_mean attr, hence we have to calculate from forcing data directly
        prcp_means = []
        for k in range(len(object_ids)):
            canopex_id = self.camels_sites[
                self.camels_sites["STATION_ID"] == "'" + object_ids[k] + "'"
            ]["CANOPEX_ID"].values[0]
            forcing_file = os.path.join(
                self.data_source_description["CAMELS_FLOW_DIR"],
                f"{str(canopex_id)}.dly",
            )
            read_forcing_file = pd.read_csv(forcing_file, header=None).values.tolist()
            prcp_data = []
            for one_site in read_forcing_file:
                all_data = one_site[0].split(" ")
                real_data = [one_data for one_data in all_data if one_data != ""]
                prcp_data.append(float(real_data[-5]))
            prcp_means.append(np.mean(np.array(prcp_data)))
        return np.array(prcp_means)
