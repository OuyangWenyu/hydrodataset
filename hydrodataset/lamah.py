"""
Author: Wenyu Ouyang
Date: 2022-01-05 18:01:11
LastEditTime: 2022-09-10 10:38:49
LastEditors: Wenyu Ouyang
Description: Read Camels datasets
FilePath: \hydrodataset\hydrodataset\lamah.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import collections
import fnmatch
import os
from typing import Union
import tarfile
import pandas as pd
import numpy as np
from pandas.core.dtypes.common import is_string_dtype, is_numeric_dtype
from tqdm import tqdm

from hydrodataset import hydro_utils
from hydrodataset.hydro_dataset import HydroDataset


class Lamah(HydroDataset):
    def __init__(self, data_path, download=False):
        """
        Initialization for LamaH-CE dataset

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
        return "LamaH_CE"

    def set_data_source_describe(self) -> collections.OrderedDict:
        """
        Introduce the files in the dataset and list their location in the file system

        Returns
        -------
        collections.OrderedDict
            the description for LamaH-CE
        """
        camels_db = self.data_source_dir

        # We use A_basins_total_upstrm
        # shp file of basins
        camels_shp_file = os.path.join(
            camels_db,
            "2_LamaH-CE_daily",
            "A_basins_total_upstrm",
            "3_shapefiles",
            "Basins_A.shp",
        )
        # config of flow data
        flow_dir = os.path.join(
            camels_db, "2_LamaH-CE_daily", "D_gauges", "2_timeseries", "daily"
        )
        forcing_dir = os.path.join(
            camels_db,
            "2_LamaH-CE_daily",
            "A_basins_total_upstrm",
            "2_timeseries",
            "daily",
        )
        attr_dir = os.path.join(
            camels_db, "2_LamaH-CE_daily", "A_basins_total_upstrm", "1_attributes"
        )

        gauge_id_file = os.path.join(
            camels_db,
            "2_LamaH-CE_daily",
            "D_gauges",
            "1_attributes",
            "Gauge_attributes.csv",
        )

        return collections.OrderedDict(
            CAMELS_DIR=camels_db,
            CAMELS_FLOW_DIR=flow_dir,
            CAMELS_FORCING_DIR=forcing_dir,
            CAMELS_ATTR_DIR=attr_dir,
            CAMELS_GAUGE_FILE=gauge_id_file,
            CAMELS_BASINS_SHP_FILE=camels_shp_file,
        )

    def download_data_source(self) -> None:
        """
        Download dataset.

        Returns
        -------
        None
        """
        camels_config = self.data_source_description
        # We only use CE's dauly files now and it is tar.gz formatting
        file = tarfile.open(
            os.path.join(camels_config["CAMELS_DIR"], "2_LamaH-CE_daily.tar.gz")
        )
        # extracting file
        file.extractall(os.path.join(camels_config["CAMELS_DIR"], "2_LamaH-CE_daily"))
        file.close()
        for f_name in os.listdir(camels_config["CAMELS_DIR"]):
            if fnmatch.fnmatch(f_name, "*.zip"):
                unzip_dir = os.path.join(camels_config["CAMELS_DIR"], f_name[0:-4])
                file_name = os.path.join(camels_config["CAMELS_DIR"], f_name)
                hydro_utils.hydro_utils.unzip_nested_zip(file_name, unzip_dir)

    def read_site_info(self) -> pd.DataFrame:
        """
        Read the basic information of gages in dataset

        Returns
        -------
        pd.DataFrame
            basic info of gages
        """
        camels_file = self.data_source_description["CAMELS_GAUGE_FILE"]
        data = pd.read_csv(camels_file, sep=";")
        return data

    def get_constant_cols(self) -> np.array:
        """
        all readable attrs

        Returns
        -------
        np.array
            attribute types
        """
        attr_all_file = os.path.join(
            self.data_source_description["CAMELS_ATTR_DIR"],
            "Catchment_attributes.csv",
        )
        lamah_ce_attr_indices_data = pd.read_csv(attr_all_file, sep=";")
        return lamah_ce_attr_indices_data.columns.values[1:]

    def get_relevant_cols(self) -> np.array:
        """
        all readable forcing types

        Returns
        -------
        np.array
            forcing types
        """
        # CE's PET data are included in "F_hydro_model" directory, here we didn't include it
        return np.array(
            [
                "2m_temp_max",
                "2m_temp_mean",
                "2m_temp_min",
                "2m_dp_temp_max",
                "2m_dp_temp_mean",
                "2m_dp_temp_min",
                "10m_wind_u",
                "10m_wind_v",
                "fcst_alb",
                "lai_high_veg",
                "lai_low_veg",
                "swe",
                "surf_net_solar_rad_max",
                "surf_net_solar_rad_mean",
                "surf_net_therm_rad_max",
                "surf_net_therm_rad_mean",
                "surf_press",
                "total_et",
                "prec",
                "volsw_123",
                "volsw_4",
            ]
        )

    def get_target_cols(self) -> np.array:
        """
        the target vars are streamflows

        Returns
        -------
        np.array
            streamflow types
        """
        return np.array(["qobs"])

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
        # Not all basins have attributes, so we just chose those with attrs
        ids = self.camels_sites["ID"].values
        attr_all_file = os.path.join(
            self.data_source_description["CAMELS_ATTR_DIR"],
            "Catchment_attributes.csv",
        )
        attr_data = pd.read_csv(attr_all_file, sep=";")
        # keep consistent with others' data type
        return np.intersect1d(ids, attr_data["ID"].values).astype(str)

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
        for k in tqdm(range(len(gage_id_lst)), desc="Read streamflow data of LamaH-CE"):
            flow_file = os.path.join(
                self.data_source_description["CAMELS_FLOW_DIR"],
                "ID_" + str(gage_id_lst[k]) + ".csv",
            )
            flow_data = pd.read_csv(flow_file, sep=";")
            df_date = flow_data[["YYYY", "MM", "DD"]]
            df_date.columns = ["year", "month", "day"]
            date = pd.to_datetime(df_date).values.astype("datetime64[D]")
            [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
            obs = flow_data["qobs"].values
            obs[obs < 0] = np.nan
            y[k, ind2, 0] = obs[ind1]
        # other units are m3/s -> ft3/s
        y = y * 35.314666721489
        return y

    def read_lamah_hydro_model_time_series(
        self, gage_id_lst: list = None, t_range: list = None, var_lst: list = None
    ):
        """
        Read time series data in hydro_model dir of Lamah dataset

        Parameters
        ----------
        gage_id_lst

        t_range

        var_lst


        Returns
        -------
        np.array

        """
        t_range_list = hydro_utils.t_range_days(t_range)
        nt = t_range_list.shape[0]
        x = np.full([len(gage_id_lst), nt, len(var_lst)], np.nan)
        all_ts_types = [
            "P_A",
            "P_B",
            "T_A",
            "T_B",
            "PET_A",
            "PET_B",
            "ETA_A",
            "ETA_B",
            "BW0_A",
            "BW0_B",
            "BW3_A",
            "BW3_B",
            "SWW_A",
            "SWW_B",
            "Qsim_A",
            "Qsim_B",
            "Qobs",
        ]
        if not set(var_lst).issubset(set(all_ts_types)):
            raise RuntimeError("No such var types")
        for k in tqdm(
            range(len(gage_id_lst)), desc="Read hydro model timeseries data of LamaH-CE"
        ):
            forcing_file = os.path.join(
                self.data_source_description["CAMELS_DIR"],
                "2_LamaH-CE_daily",
                "F_hydrol_model",
                "2_timeseries",
                "ID_" + str(gage_id_lst[k]) + ".csv",
            )
            forcing_data = pd.read_csv(forcing_file, sep=";")
            df_date = forcing_data[["YYYY", "MM", "DD"]]
            df_date.columns = ["year", "month", "day"]
            date = pd.to_datetime(df_date).values.astype("datetime64[D]")
            [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
            for j in range(len(var_lst)):
                if "PET" in var_lst[j]:
                    # pet value may be negative
                    pet = forcing_data[var_lst[j]].values
                    pet[pet < 0] = 0.0
                    x[k, ind2, j] = pet[ind1]
                else:
                    x[k, ind2, j] = forcing_data[var_lst[j]].values[ind1]
        return x

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
        for k in tqdm(range(len(gage_id_lst)), desc="Read forcing data of LamaH-CE"):
            forcing_file = os.path.join(
                self.data_source_description["CAMELS_FORCING_DIR"],
                "ID_" + str(gage_id_lst[k]) + ".csv",
            )
            forcing_data = pd.read_csv(forcing_file, sep=";")
            df_date = forcing_data[["YYYY", "MM", "DD"]]
            df_date.columns = ["year", "month", "day"]
            date = pd.to_datetime(df_date).values.astype("datetime64[D]")
            [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
            for j in range(len(var_lst)):
                x[k, ind2, j] = forcing_data[var_lst[j]].values[ind1]
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
            "Catchment_attributes.csv",
        )
        all_attr = pd.read_csv(attr_all_file, sep=";")
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
        return self.read_constant_cols(object_ids, ["area_calc"], is_return_dict=False)

    def read_mean_prep(self, object_ids) -> np.array:
        return self.read_constant_cols(object_ids, ["p_mean"], is_return_dict=False)
