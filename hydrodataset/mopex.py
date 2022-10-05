"""
Author: Wenyu Ouyang
Date: 2022-01-05 18:01:11
LastEditTime: 2022-09-10 10:38:08
LastEditors: Wenyu Ouyang
Description: Read Camels datasets
FilePath: \hydrodataset\hydrodataset\mopex.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import collections
import fnmatch
import os
from typing import Union
import pandas as pd
import numpy as np
from pandas.core.dtypes.common import is_string_dtype, is_numeric_dtype
from tqdm import tqdm

from hydrodataset.hydro_dataset import HydroDataset
from hydrodataset import hydro_utils


class Mopex(HydroDataset):
    def __init__(self, data_path, download=False):
        """
        Initialization for dataset

        Parameters
        ----------
        data_path
            where we put the dataset
        download
            if true, download
        """
        super().__init__(data_path)
        self.data_source_description = self.set_data_source_describe()
        if download:
            self.download_data_source()
        self.camels_sites = self.read_site_info()

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
                unzip_dir = os.path.join(camels_config["CAMELS_DIR"], f_name[0:-4])
                file_name = os.path.join(camels_config["CAMELS_DIR"], f_name)
                hydro_utils.unzip_nested_zip(file_name, unzip_dir)

    def read_site_info(self) -> pd.DataFrame:
        """
        Read the basic information of gages in dataset

        Returns
        -------
        pd.DataFrame
            basic info of gages
        """
        camels_file = self.data_source_description["CAMELS_GAUGE_FILE"]
        data = pd.read_excel(camels_file)
        return data

    def get_constant_cols(self) -> np.array:
        """
        all readable attrs

        Returns
        -------
        np.array
            attribute types
        """
        data_folder = self.data_source_description["CAMELS_ATTR_DIR"]
        attr_all_file = os.path.join(
            self.data_source_description["CAMELS_DIR"],
            "HYSETS_watershed_properties.txt",
        )
        canopex_attr_indices_data = pd.read_csv(attr_all_file, sep=";")
        # exclude HYSETS watershed id
        return canopex_attr_indices_data.columns.values[1:]

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

    def read_target_cols(
        self,
        gage_id_lst: Union[list, np.array] = None,
        t_range: list = None,
        target_cols: Union[list, np.array] = None,
        **kwargs
    ) -> np.array:
        """
        read target values; they are streamflows

        default target_cols is an one-value list
        Notice: the unit of target outputs in different regions are not totally same

        Parameters
        ----------
        gage_id_lst
            station ids
        t_range
            the time range, for example, ["1990-01-01", "2000-01-01"]
        target_cols
            the default is None, but we neea at least one default target.
            For CAMELS-US, it is ["usgsFlow"];
            for CAMELS-AUS, it's ["streamflow_mmd"]
            for CAMELS-AUS, it's ["streamflow_m3s"]
        kwargs
            some other params if needed

        Returns
        -------
        np.array
            streamflow data, 3-dim [station, time, streamflow]
        """
        if target_cols is None:
            return np.array([])
        else:
            nf = len(target_cols)
        t_range_list = hydro_utils.t_range_days(t_range)
        nt = t_range_list.shape[0]
        y = np.full([len(gage_id_lst), nt, nf], np.nan)

        for k in tqdm(range(len(gage_id_lst)), desc="Read streamflow data of CANOPEX"):
            # only one streamflow type: discharge
            canopex_id = self.camels_sites[
                self.camels_sites["STATION_ID"] == "'" + gage_id_lst[k] + "'"
            ]["CANOPEX_ID"].values[0]
            flow_file = os.path.join(
                self.data_source_description["CAMELS_FLOW_DIR"],
                str(canopex_id) + ".dly",
            )
            read_flow_file = pd.read_csv(flow_file, header=None).values.tolist()
            flow_data = []
            flow_date = []
            for one_site in read_flow_file:
                flow_date.append(
                    hydro_utils.t2dt(int(one_site[0][:8].replace(" ", "0")))
                )
                all_data = one_site[0].split(" ")
                real_data = [one_data for one_data in all_data if one_data != ""]
                flow_data.append(float(real_data[-3]))
            date = pd.to_datetime(flow_date).values.astype("datetime64[D]")
            [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
            obs = np.array(flow_data)
            obs[obs < 0] = np.nan
            y[k, ind2, 0] = obs[ind1]
        # other units are m3/s -> ft3/s
        y = y * 35.314666721489
        return y

    def read_relevant_cols(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        forcing_type="daymet",
    ) -> np.array:
        """
        Read forcing data

        Parameters
        ----------
        gage_id_lst
            station ids
        t_range
            the time range, for example, ["1990-01-01", "2000-01-01"]
        var_lst
            forcing variable types
        forcing_type
            now only for CAMELS-US, there are three types: daymet, nldas, maurer
        Returns
        -------
        np.array
            forcing data
        """
        t_range_list = hydro_utils.t_range_days(t_range)
        nt = t_range_list.shape[0]
        x = np.full([len(gage_id_lst), nt, len(var_lst)], np.nan)

        for k in tqdm(range(len(gage_id_lst)), desc="Read forcing data of CANOPEX"):
            canopex_id = self.camels_sites[
                self.camels_sites["STATION_ID"] == "'" + gage_id_lst[k] + "'"
            ]["CANOPEX_ID"].values[0]
            forcing_file = os.path.join(
                self.data_source_description["CAMELS_FLOW_DIR"],
                str(canopex_id) + ".dly",
            )
            read_forcing_file = pd.read_csv(forcing_file, header=None).values.tolist()

            forcing_date = []
            for j in range(len(var_lst)):
                forcing_data = []
                for one_site in read_forcing_file:
                    forcing_date.append(
                        hydro_utils.t2dt(int(one_site[0][:8].replace(" ", "0")))
                    )
                    all_data = one_site[0].split(" ")
                    real_data = [one_data for one_data in all_data if one_data != ""]
                    if var_lst[j] == "prcp":
                        forcing_data.append(float(real_data[-5]))
                    elif var_lst[j] == "tmax":
                        forcing_data.append(float(real_data[-2]))
                    elif var_lst[j] == "tmin":
                        forcing_data.append(float(real_data[-1]))
                    else:
                        raise NotImplementedError(
                            "No such forcing type in CANOPEX now!"
                        )
                date = pd.to_datetime(forcing_date).values.astype("datetime64[D]")
                [c, ind1, ind2] = np.intersect1d(
                    date, t_range_list, return_indices=True
                )
                x[k, ind2, j] = np.array(forcing_data)[ind1]
        return x

    def read_attr_all_in_one_file(self):
        """
        Read all attr data

        Returns
        -------
        np.array
            all attr data
        """

        attr_all_file = os.path.join(
            self.data_source_description["CAMELS_ATTR_DIR"],
            "HYSETS_watershed_properties.txt",
        )
        all_attr_tmp = pd.read_csv(attr_all_file, sep=";", index_col=0)
        all_attr = all_attr_tmp[
            all_attr_tmp["Official_ID"].isin(self.read_object_ids())
        ]
        # gage_all_attr = all_attr[all_attr['station_id'].isin(gage_id_lst)]
        var_lst = self.get_constant_cols().tolist()
        data_temp = all_attr[var_lst]
        # for factorized data, we need factorize all gages' data to keep the factorized number same all the time
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
        # keep same format with CAMELS_US
        return out, var_lst, None, f_dict

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
        attr_all, var_lst_all, var_dict, f_dict = self.read_attr_all_in_one_file()
        ind_var = [var_lst_all.index(var) for var in var_lst]
        id_lst_all = self.read_object_ids()
        # Notice the sequence of station ids ! Some id_lst_all are not sorted, so don't use np.intersect1d
        ind_grid = [id_lst_all.tolist().index(tmp) for tmp in gage_id_lst]
        temp = attr_all[ind_grid, :]
        out = temp[:, ind_var]
        if is_return_dict:
            return out, var_dict, f_dict
        else:
            return out

    def read_basin_area(self, object_ids) -> np.array:
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
                str(canopex_id) + ".dly",
            )
            read_forcing_file = pd.read_csv(forcing_file, header=None).values.tolist()
            prcp_data = []
            for one_site in read_forcing_file:
                all_data = one_site[0].split(" ")
                real_data = [one_data for one_data in all_data if one_data != ""]
                prcp_data.append(float(real_data[-5]))
            prcp_means.append(np.mean(np.array(prcp_data)))
        return np.array(prcp_means)
