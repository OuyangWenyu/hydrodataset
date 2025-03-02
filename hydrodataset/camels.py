"""
Author: Wenyu Ouyang
Date: 2022-01-05 18:01:11
LastEditTime: 2024-11-11 17:26:42
LastEditors: Wenyu Ouyang
Description: Read Camels Series ("UnitedStates") datasets
FilePath: \hydrodataset\hydrodataset\camels.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import json
import warnings
import collections
import fnmatch
import logging
import os
from typing import Union
import pandas as pd
import numpy as np
from pandas.core.dtypes.common import is_string_dtype, is_numeric_dtype
from pathlib import Path
from urllib.request import urlopen
from tqdm import tqdm
import xarray as xr
from hydroutils import hydro_time, hydro_file
from hydrodataset import CACHE_DIR, HydroDataset, CAMELS_REGIONS

CAMELS_NO_DATASET_ERROR_LOG = (
    "We cannot read this dataset now. Please check if you choose correctly:\n"
    + str(CAMELS_REGIONS)
)


def map_string_vars(ds):
    # Iterate over all variables in the dataset
    for var in ds.data_vars:
        # Check if the variable contains string data
        if ds[var].dtype == object:
            # Convert the DataArray to a pandas Series
            var_series = ds[var].to_series()

            # Get all unique strings and create a mapping to integers
            unique_strings = sorted(var_series.unique())
            mapping = {value: i for i, value in enumerate(unique_strings)}

            # Apply the mapping to the series
            mapped_series = var_series.map(mapping)

            # Convert the series back to a DataArray and replace the old one in the Dataset
            ds[var] = xr.DataArray(mapped_series)

    return ds


def time_intersect_dynamic_data(obs: np.array, date: np.array, t_range: list):
    """
    chose data from obs in the t_range

    Parameters
    ----------
    obs
        a np array
    date
        all periods for obs
    t_range
        the time range we need, such as ["1990-01-01","2000-01-01"]

    Returns
    -------
    np.array
        the chosen data
    """
    t_lst = hydro_time.t_range_days(t_range)
    nt = t_lst.shape[0]
    if len(obs) != nt:
        out = np.full([nt], np.nan)
        [c, ind1, ind2] = np.intersect1d(date, t_lst, return_indices=True)
        out[ind2] = obs[ind1]
    else:
        out = obs
    return out


class Camels(HydroDataset):
    def __init__(
        self,
        data_path=os.path.join("camels", "camels_us"),
        download=False,
        region: str = "US",
    ):
        """
        Initialization for CAMELS series dataset

        Parameters
        ----------
        data_path
            where we put the dataset.
            we already set the ROOT directory for hydrodataset,
            so here just set it as a relative path,
            by default "camels/camels_us"
        download
            if true, download, by defaulf False
        region
            the default is CAMELS(-US), since it's the first CAMELS dataset.
            All are included in CAMELS_REGIONS
        """
        super().__init__(data_path)
        if region not in CAMELS_REGIONS:
            raise NotImplementedError(
                f"Please chose one region in: {str(CAMELS_REGIONS)}"
            )
        self.region = region
        self.data_source_description = self.set_data_source_describe()
        if download:
            self.download_data_source()
        self.sites = self.read_site_info()

    def get_name(self):
        return "CAMELS_" + self.region

    def set_data_source_describe(self) -> collections.OrderedDict:
        """
        the files in the dataset and their location in file system

        Returns
        -------
        collections.OrderedDict
            the description for a CAMELS dataset
        """
        camels_db = self.data_source_dir

        return self._set_data_source_camelsus_describe(camels_db)

    def _set_data_source_camelsus_describe(self, camels_db):
        # shp file of basins
        camels_shp_file = camels_db.joinpath(
            "basin_set_full_res", "HCDN_nhru_final_671.shp"
        )
        # config of flow data
        flow_dir = camels_db.joinpath(
            "basin_timeseries_v1p2_metForcing_obsFlow",
            "basin_dataset_public_v1p2",
            "usgs_streamflow",
        )
        flow_after_2015_dir = camels_db.joinpath(
            "camels_streamflow", "camels_streamflow"
        )
        # forcing
        forcing_dir = camels_db.joinpath(
            "basin_timeseries_v1p2_metForcing_obsFlow",
            "basin_dataset_public_v1p2",
            "basin_mean_forcing",
        )
        forcing_types = ["daymet", "maurer", "nldas"]
        # attr
        attr_dir = camels_db
        gauge_id_file = attr_dir.joinpath("camels_name.txt")
        attr_key_lst = ["topo", "clim", "hydro", "vege", "soil", "geol"]
        base_url = "https://gdex.ucar.edu/dataset/camels"
        download_url_lst = [
            f"{base_url}/file/basin_set_full_res.zip",
            # f"{base_url}/file/basin_timeseries_v1p2_metForcing_obsFlow.zip",
            f"{base_url}/file/camels_attributes_v2.0.xlsx",
            f"{base_url}/file/camels_clim.txt",
            f"{base_url}/file/camels_geol.txt",
            f"{base_url}/file/camels_hydro.txt",
            f"{base_url}/file/camels_name.txt",
            f"{base_url}/file/camels_soil.txt",
            f"{base_url}/file/camels_topo.txt",
            f"{base_url}/file/camels_vege.txt",
        ]

        return collections.OrderedDict(
            CAMELS_DIR=camels_db,
            CAMELS_FLOW_DIR=flow_dir,
            CAMELS_FLOW_AFTER2015_DIR=flow_after_2015_dir,
            CAMELS_FORCING_DIR=forcing_dir,
            CAMELS_FORCING_TYPE=forcing_types,
            CAMELS_ATTR_DIR=attr_dir,
            CAMELS_ATTR_KEY_LST=attr_key_lst,
            CAMELS_GAUGE_FILE=gauge_id_file,
            CAMELS_BASINS_SHP_FILE=camels_shp_file,
            CAMELS_DOWNLOAD_URL_LST=download_url_lst,
        )

    def download_data_source(self) -> None:
        """
        Download the required zip files

        Now we only support CAMELS-US's downloading.
        For others, please download it manually,
        and put all files of a dataset in one directory.
        For example, all files of CAMELS_AUS should be put in "camels_aus"

        Returns
        -------
        None
        """
        camels_config = self.data_source_description

        self.data_source_dir.mkdir(exist_ok=True)
        links = camels_config["CAMELS_DOWNLOAD_URL_LST"]
        for url in links:
            fzip = Path(self.data_source_dir, url.rsplit("/", 1)[1])
            if fzip.exists():
                with urlopen(url) as response:
                    if (
                        int(response.info()["Content-length"])
                        != fzip.stat().st_size
                    ):
                        fzip.unlink()
        to_dl = [
            url
            for url in links
            if not Path(self.data_source_dir, url.rsplit("/", 1)[1]).exists()
        ]
        hydro_file.download_zip_files(to_dl, self.data_source_dir)

        hydro_file.zip_extract(camels_config["CAMELS_DIR"])

    def read_site_info(self) -> pd.DataFrame:
        """
        Read the basic information of gages in a CAMELS dataset

        Returns
        -------
        pd.DataFrame
            basic info of gages
        """
        camels_file = self.data_source_description["CAMELS_GAUGE_FILE"]
        data = pd.read_csv(
            camels_file, sep=";", dtype={"gauge_id": str, "huc_02": str}
        )
        return data

    def get_constant_cols(self) -> np.ndarray:
        """
        all readable attrs in CAMELS

        Returns
        -------
        np.array
            attribute types
        """
        data_folder = self.data_source_description["CAMELS_ATTR_DIR"]
        return self._get_constant_cols_some(data_folder, "camels_", ".txt", ";")

    def _get_constant_cols_some(self, data_folder, arg1, arg2, sep):
        var_dict = {}
        var_lst = []
        key_lst = self.data_source_description["CAMELS_ATTR_KEY_LST"]
        for key in key_lst:
            data_file = os.path.join(data_folder, arg1 + key + arg2)
            data_temp = pd.read_csv(data_file, sep=sep)
            var_lst_temp = list(data_temp.columns[1:])
            var_dict[key] = var_lst_temp
            var_lst.extend(var_lst_temp)
        return np.array(var_lst)

    def get_relevant_cols(self) -> np.ndarray:
        """
        all readable forcing types

        Returns
        -------
        np.array
            forcing types
        """

        # PET is from model_output file in CAMELS-US
        return np.array(
            ["dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp", "PET"]
        )

    def get_target_cols(self) -> np.ndarray:
        """
        For CAMELS, the target vars are streamflows

        Returns
        -------
        np.array
            streamflow types
        """
        return np.array(["usgsFlow", "ET"])

    def read_object_ids(self, **kwargs) -> np.ndarray:
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

    def read_usgs_gage(self, usgs_id, t_range):
        """
        read streamflow data of a station for date before 2015-01-01 from CAMELS-US

        Parameters
        ----------
        usgs_id
            the station id
        t_range
            the time range, for example, ["1990-01-01", "2000-01-01"]

        Returns
        -------
        np.array
            streamflow data of one station for a given time range
        """
        logging.debug("reading %s streamflow data before 2015", usgs_id)
        gage_id_df = self.sites
        huc = gage_id_df[gage_id_df["gauge_id"] == usgs_id]["huc_02"].values[0]
        usgs_file = os.path.join(
            self.data_source_description["CAMELS_FLOW_DIR"],
            huc,
            usgs_id + "_streamflow_qc.txt",
        )
        data_temp = pd.read_csv(usgs_file, sep=r"\s+", header=None)
        obs = data_temp[4].values
        obs[obs < 0] = np.nan
        t_lst = hydro_time.t_range_days(t_range)
        nt = t_lst.shape[0]
        return (
            self._read_usgs_gage_for_some(nt, data_temp, t_lst, obs)
            if len(obs) != nt
            else obs
        )

    def _read_usgs_gage_for_some(self, nt, data_temp, t_lst, obs):
        result = np.full([nt], np.nan)
        df_date = data_temp[[1, 2, 3]]
        df_date.columns = ["year", "month", "day"]
        date = pd.to_datetime(df_date).values.astype("datetime64[D]")
        [C, ind1, ind2] = np.intersect1d(date, t_lst, return_indices=True)
        result[ind2] = obs[ind1]
        return result

    def read_camels_streamflow(self, usgs_id, t_range):
        """
        read streamflow data of a station for date after 2015 from CAMELS-US

        The streamflow data is downloaded from USGS website by HyRivers tools

        Parameters
        ----------
        usgs_id
            the station id
        t_range
            the time range, for example, ["2015-01-01", "2022-01-01"]

        Returns
        -------
        np.array
            streamflow data of one station for a given time range
        """
        logging.debug("reading %s streamflow data after 2015", usgs_id)
        gage_id_df = self.sites
        huc = gage_id_df[gage_id_df["gauge_id"] == usgs_id]["huc_02"].values[0]
        usgs_file = os.path.join(
            self.data_source_description["CAMELS_FLOW_AFTER2015_DIR"],
            huc,
            usgs_id + "_streamflow_qc.txt",
        )
        data_temp = pd.read_csv(usgs_file, sep=",", header=None, skiprows=1)
        obs = data_temp[4].values
        obs[obs < 0] = np.nan
        t_lst = hydro_time.t_range_days(t_range)
        nt = t_lst.shape[0]
        return (
            self._read_usgs_gage_for_some(nt, data_temp, t_lst, obs)
            if len(obs) != nt
            else obs
        )

    def read_target_cols(
        self,
        gage_id_lst: Union[list, np.array] = None,
        t_range: list = None,
        target_cols: Union[list, np.array] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        read target values; for CAMELS, they are streamflows

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
        t_range_list = hydro_time.t_range_days(t_range)
        nt = t_range_list.shape[0]
        y = np.full([len(gage_id_lst), nt, nf], np.nan)
        for k in tqdm(
            range(len(gage_id_lst)), desc="Read streamflow data of CAMELS-US"
        ):
            for j in range(len(target_cols)):
                if target_cols[j] == "ET":
                    data_et = self.read_camels_us_model_output_data(
                        gage_id_lst[k : k + 1], t_range, ["ET"]
                    )
                    y[k, :, j : j + 1] = data_et
                else:
                    data_obs = self._read_augmented_camels_streamflow(
                        gage_id_lst, t_range, t_range_list, k
                    )
                    y[k, :, j] = data_obs

        # Keep unit of streamflow unified: we use ft3/s here
        return y

    def _read_augmented_camels_streamflow(self, gage_id_lst, t_range, t_range_list, k):
        dt150101 = hydro_time.t2str("2015-01-01")
        if t_range_list[-1] > dt150101 and t_range_list[0] < dt150101:
            # latest streamflow data in CAMELS is 2014/12/31
            data_obs_after_2015 = self.read_camels_streamflow(
                gage_id_lst[k], ["2015-01-01", t_range[1]]
            )
            data_obs_before_2015 = self.read_usgs_gage(
                gage_id_lst[k], [t_range[0], "2015-01-01"]
            )
            return np.concatenate((data_obs_before_2015, data_obs_after_2015))
        elif t_range_list[-1] <= dt150101:
            return self.read_usgs_gage(gage_id_lst[k], t_range)
        else:
            return self.read_camels_streamflow(gage_id_lst[k], t_range)

    def read_camels_us_model_output_data(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        forcing_type="daymet",
    ) -> np.array:
        """
        Read model output data of CAMELS-US, including SWE, PRCP, RAIM, TAIR, PET, ET, MOD_RUN, OBS_RUN
        Date starts from 1980-10-01 to 2014-12-31

        Parameters
        ----------
        gage_id_lst : list
            the station id list
        var_lst : list
            the variable list
        t_range : list
            the time range, for example, ["1990-01-01", "2000-01-01"]
        forcing_type : str, optional
            by default "daymet"
        """
        t_range_list = hydro_time.t_range_days(t_range)
        model_out_put_var_lst = [
            "SWE",
            "PRCP",
            "RAIM",
            "TAIR",
            "PET",
            "ET",
            "MOD_RUN",
            "OBS_RUN",
        ]
        if not set(var_lst).issubset(set(model_out_put_var_lst)):
            raise RuntimeError("not in this list")
        nt = t_range_list.shape[0]
        chosen_camels_mods = np.full([len(gage_id_lst), nt, len(var_lst)], np.nan)
        count = 0
        for usgs_id in tqdm(gage_id_lst, desc="Read model output data of CAMELS-US"):
            gage_id_df = self.sites
            huc02_ = gage_id_df[gage_id_df["gauge_id"] == usgs_id]["huc_02"].values[0]
            file_path_dir = os.path.join(
                self.data_source_dir,
                "basin_timeseries_v1p2_modelOutput_" + forcing_type,
                "model_output_" + forcing_type,
                "model_output",
                "flow_timeseries",
                forcing_type,
                huc02_,
            )
            sac_random_seeds = [
                "05",
                "11",
                "27",
                "33",
                "48",
                "59",
                "66",
                "72",
                "80",
                "94",
            ]
            files = [
                os.path.join(
                    file_path_dir, usgs_id + "_" + random_seed + "_model_output.txt"
                )
                for random_seed in sac_random_seeds
            ]
            results = []
            for file in files:
                result = pd.read_csv(file, sep="\s+")
                df_date = result[["YR", "MNTH", "DY"]]
                df_date.columns = ["year", "month", "day"]
                date = pd.to_datetime(df_date).values.astype("datetime64[D]")
                [c, ind1, ind2] = np.intersect1d(
                    date, t_range_list, return_indices=True
                )
                results.append(result[var_lst].values[ind1])
            result_np = np.array(results)
            chosen_camels_mods[count, ind2, :] = np.mean(result_np, axis=0)
            count = count + 1
        return chosen_camels_mods

    def read_forcing_gage(self, usgs_id, var_lst, t_range_list, forcing_type="daymet"):
        # data_source = daymet or maurer or nldas
        logging.debug("reading %s forcing data", usgs_id)
        gage_id_df = self.sites
        huc = gage_id_df[gage_id_df["gauge_id"] == usgs_id]["huc_02"].values[0]

        data_folder = self.data_source_description["CAMELS_FORCING_DIR"]
        temp_s = "cida" if forcing_type == "daymet" else forcing_type
        data_file = os.path.join(
            data_folder,
            forcing_type,
            huc,
            f"{usgs_id}_lump_{temp_s}_forcing_leap.txt",
        )
        data_temp = pd.read_csv(data_file, sep=r"\s+", header=None, skiprows=4)
        forcing_lst = [
            "Year",
            "Mnth",
            "Day",
            "Hr",
            "dayl",
            "prcp",
            "srad",
            "swe",
            "tmax",
            "tmin",
            "vp",
        ]
        df_date = data_temp[[0, 1, 2]]
        df_date.columns = ["year", "month", "day"]
        date = pd.to_datetime(df_date).values.astype("datetime64[D]")

        nf = len(var_lst)
        [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
        nt = c.shape[0]
        out = np.full([nt, nf], np.nan)

        for k in range(nf):
            ind = forcing_lst.index(var_lst[k])
            out[ind2, k] = data_temp[ind].values[ind1]
        return out

    def read_relevant_cols(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        forcing_type="daymet",
    ) -> np.ndarray:
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
        t_range_list = hydro_time.t_range_days(t_range)
        nt = t_range_list.shape[0]
        x = np.full([len(gage_id_lst), nt, len(var_lst)], np.nan)
        for k in tqdm(
            range(len(gage_id_lst)), desc="Read forcing data of CAMELS-US"
        ):
            if "PET" in var_lst:
                pet_idx = var_lst.index("PET")
                data_pet = self.read_camels_us_model_output_data(
                    gage_id_lst[k : k + 1], t_range, ["PET"]
                )
                x[k, :, pet_idx : pet_idx + 1] = data_pet
                no_pet_var_lst = [x for x in var_lst if x != "PET"]
                data = self.read_forcing_gage(
                    gage_id_lst[k],
                    no_pet_var_lst,
                    t_range_list,
                    forcing_type=forcing_type,
                )
                var_indices = [var_lst.index(var) for var in no_pet_var_lst]
                x[k : k + 1, :, var_indices] = data
            else:
                data = self.read_forcing_gage(
                    gage_id_lst[k],
                    var_lst,
                    t_range_list,
                    forcing_type=forcing_type,
                )
                x[k, :, :] = data
        return x

    def read_attr_all(self):
        data_folder = self.data_source_description["CAMELS_ATTR_DIR"]
        key_lst = self.data_source_description["CAMELS_ATTR_KEY_LST"]
        f_dict = {}
        var_dict = {}
        var_lst = []
        out_lst = []
        gage_dict = self.sites
        camels_str = "camels_"
        sep_ = ";"
        for key in key_lst:
            data_file = os.path.join(data_folder, camels_str + key + ".txt")
            if self.region == "GB":
                data_file = os.path.join(
                    data_folder, camels_str + key + "_attributes.csv"
                )
            elif self.region == "CC":
                data_file = os.path.join(data_folder, key + ".csv")
            data_temp = pd.read_csv(data_file, sep=sep_)
            var_lst_temp = list(data_temp.columns[1:])
            var_dict[key] = var_lst_temp
            var_lst.extend(var_lst_temp)
            k = 0
            gage_id_key = "gauge_id"
            if self.region == "CC":
                gage_id_key = "gage_id"
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

    def read_attr_all_yr(self):
        var_lst = self.get_constant_cols().tolist()
        gage_id_lst = self.read_object_ids()
        # for factorized data, we need factorize all gages' data to keep the factorized number same all the time
        n_gage = len(self.read_object_ids())
        c = np.full([n_gage, len(var_lst)], np.nan, dtype=object)
        for k in range(n_gage):
            attr_file = os.path.join(
                self.data_source_description["CAMELS_ATTR_DIR"],
                gage_id_lst[k],
                "attributes.json",
            )
            attr_data = hydro_file.unserialize_json_ordered(attr_file)
            for j in range(len(var_lst)):
                c[k, j] = attr_data[var_lst[j]]
        data_temp = pd.DataFrame(c, columns=var_lst)
        out_temp = np.full([n_gage, len(var_lst)], np.nan)
        f_dict = {}
        k = 0
        for field in var_lst:
            if field in ["high_prec_timing", "low_prec_timing"]:
                # string type
                value, ref = pd.factorize(data_temp[field], sort=True)
                out_temp[:, k] = value
                f_dict[field] = ref.tolist()
            else:
                out_temp[:, k] = data_temp[field].values
            k = k + 1
        # keep same format with CAMELS_US
        return out_temp, var_lst, None, f_dict

    def read_constant_cols(
        self, gage_id_lst=None, var_lst=None, is_return_dict=False
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
            if true, return var_dict and f_dict for CAMELS-US
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
        return self.read_attr_xrdataset(gage_id_lst, ["area_gages2"])


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


        Raises
        ------
        NotImplementedError
            some regions are not supported
        ValueError
            unit must be one of ['mm/d', 'mm/day', 'mm/h', 'mm/hour', 'mm/3h', 'mm/3hour', 'mm/8d', 'mm/8day']
        """
        data = self.read_attr_xrdataset(
            gage_id_lst,
            ["p_mean"],
            is_return_dict=False,
        )
        converted_data = self.unit_convert_mean_prcp(data)
        return converted_data

    def cache_forcing_np_json(self):
        """
        Save all daymet basin-forcing data in a numpy array file in the cache directory.

        Because it takes much time to read data from txt files,
        it is a good way to cache data as a numpy file to speed up the reading.
        In addition, we need a document to explain the meaning of all dimensions.

        """
        cache_npy_file = CACHE_DIR.joinpath("camels_daymet_forcing.npy")
        json_file = CACHE_DIR.joinpath("camels_daymet_forcing.json")
        variables = self.get_relevant_cols()
        basins = self.sites["gauge_id"].values
        daymet_t_range = ["1980-01-01", "2015-01-01"]
        times = [
            hydro_time.t2str(tmp)
            for tmp in hydro_time.t_range_days(daymet_t_range).tolist()
        ]
        data_info = collections.OrderedDict(
            {
                "dim": ["basin", "time", "variable"],
                "basin": basins.tolist(),
                "time": times,
                "variable": variables.tolist(),
            }
        )
        with open(json_file, "w") as FP:
            json.dump(data_info, FP, indent=4)
        data = self.read_relevant_cols(
            gage_id_lst=basins.tolist(),
            t_range=daymet_t_range,
            var_lst=variables.tolist(),
        )
        np.save(cache_npy_file, data)

    def cache_streamflow_np_json(self):
        """
        Save all basins' streamflow data in a numpy array file in the cache directory

        """
        cache_npy_file = CACHE_DIR.joinpath("camels_streamflow.npy")
        json_file = CACHE_DIR.joinpath("camels_streamflow.json")
        variables = self.get_target_cols()
        basins = self.sites["gauge_id"].values
        t_range = ["1980-01-01", "2015-01-01"]
        times = [
            hydro_time.t2str(tmp) for tmp in hydro_time.t_range_days(t_range).tolist()
        ]
        data_info = collections.OrderedDict(
            {
                "dim": ["basin", "time", "variable"],
                "basin": basins.tolist(),
                "time": times,
                "variable": variables.tolist(),
            }
        )
        with open(json_file, "w") as FP:
            json.dump(data_info, FP, indent=4)
        data = self.read_target_cols(
            gage_id_lst=basins,
            t_range=t_range,
            target_cols=variables,
        )
        np.save(cache_npy_file, data)

    def cache_attributes_xrdataset(self):
        """Convert all the attributes to a single dataframe

        Returns
        -------
        None
        """
        # NOTICE: although it seems that we don't use pint_xarray, we have to import this package
        import pint_xarray

        attr_files = self.data_source_dir.glob("camels_*.txt")
        attrs = {
            f.stem.split("_")[1]: pd.read_csv(
                f, sep=";", index_col=0, dtype={"huc_02": str, "gauge_id": str}
            )
            for f in attr_files
        }

        attrs_df = pd.concat(attrs.values(), axis=1)

        # fix station names
        def fix_station_nm(station_nm):
            name = station_nm.title().rsplit(" ", 1)
            name[0] = name[0] if name[0][-1] == "," else f"{name[0]},"
            name[1] = name[1].replace(".", "")
            return " ".join(
                (name[0], name[1].upper() if len(name[1]) == 2 else name[1].title())
            )

        attrs_df["gauge_name"] = [fix_station_nm(n) for n in attrs_df["gauge_name"]]
        obj_cols = attrs_df.columns[attrs_df.dtypes == "object"]
        for c in obj_cols:
            attrs_df[c] = attrs_df[c].str.strip().astype(str)

        # transform categorical variables to numeric
        categorical_mappings = {}
        for column in attrs_df.columns:
            if attrs_df[column].dtype == "object":
                attrs_df[column] = attrs_df[column].astype("category")
                categorical_mappings[column] = dict(
                    enumerate(attrs_df[column].cat.categories)
                )
                attrs_df[column] = attrs_df[column].cat.codes

        # unify id to basin
        attrs_df.index.name = "basin"
        # We use xarray dataset to cache all data
        ds_from_df = attrs_df.to_xarray()
        units_dict = {
            "gauge_lat": "degree",
            "gauge_lon": "degree",
            "elev_mean": "m",
            "slope_mean": "m/km",
            "area_gages2": "km^2",
            "area_geospa_fabric": "km^2",
            "geol_1st_class": "dimensionless",
            "glim_1st_class_frac": "dimensionless",
            "geol_2nd_class": "dimensionless",
            "glim_2nd_class_frac": "dimensionless",
            "carbonate_rocks_frac": "dimensionless",
            "geol_porostiy": "dimensionless",
            "geol_permeability": "m^2",
            "frac_forest": "dimensionless",
            "lai_max": "dimensionless",
            "lai_diff": "dimensionless",
            "gvf_max": "dimensionless",
            "gvf_diff": "dimensionless",
            "dom_land_cover_frac": "dimensionless",
            "dom_land_cover": "dimensionless",
            "root_depth_50": "m",
            "root_depth_99": "m",
            "q_mean": "mm/day",
            "runoff_ratio": "dimensionless",
            "slope_fdc": "dimensionless",
            "baseflow_index": "dimensionless",
            "stream_elas": "dimensionless",
            "q5": "mm/day",
            "q95": "mm/day",
            "high_q_freq": "day/year",
            "high_q_dur": "day",
            "low_q_freq": "day/year",
            "low_q_dur": "day",
            "zero_q_freq": "percent",
            "hfd_mean": "dimensionless",
            "soil_depth_pelletier": "m",
            "soil_depth_statsgo": "m",
            "soil_porosity": "dimensionless",
            "soil_conductivity": "cm/hr",
            "max_water_content": "m",
            "sand_frac": "percent",
            "silt_frac": "percent",
            "clay_frac": "percent",
            "water_frac": "percent",
            "organic_frac": "percent",
            "other_frac": "percent",
            "p_mean": "mm/day",
            "pet_mean": "mm/day",
            "p_seasonality": "dimensionless",
            "frac_snow": "dimensionless",
            "aridity": "dimensionless",
            "high_prec_freq": "days/year",
            "high_prec_dur": "day",
            "high_prec_timing": "dimensionless",
            "low_prec_freq": "days/year",
            "low_prec_dur": "day",
            "low_prec_timing": "dimensionless",
            "huc_02": "dimensionless",
            "gauge_name": "dimensionless",
        }

        # Assign units to the variables in the Dataset
        for var_name in units_dict:
            if var_name in ds_from_df.data_vars:
                ds_from_df[var_name].attrs["units"] = units_dict[var_name]

        # Assign categorical mappings to the variables in the Dataset
        for column in ds_from_df.data_vars:
            if column in categorical_mappings:
                mapping_str = categorical_mappings[column]
                ds_from_df[column].attrs["category_mapping"] = str(mapping_str)
        return ds_from_df

    def cache_streamflow_xrdataset(self):
        """Save all basins' streamflow data in a netcdf file in the cache directory

        """
        cache_npy_file = CACHE_DIR.joinpath("camels_streamflow.npy")
        json_file = CACHE_DIR.joinpath("camels_streamflow.json")
        if (not os.path.isfile(cache_npy_file)) or (not os.path.isfile(json_file)):
            self.cache_streamflow_np_json()
        streamflow = np.load(cache_npy_file)
        with open(json_file, "r") as fp:
            streamflow_dict = json.load(fp, object_pairs_hook=collections.OrderedDict)
        import pint_xarray

        basins = streamflow_dict["basin"]
        times = pd.date_range(
            streamflow_dict["time"][0], periods=len(streamflow_dict["time"])
        )
        return xr.Dataset(
            {
                "streamflow": (
                    ["basin", "time"],
                    streamflow[:, :, 0],
                    {"units": self.streamflow_unit},
                ),
                "ET": (
                    ["basin", "time"],
                    streamflow[:, :, 1],
                    {"units": "mm/day"},
                ),
            },
            coords={
                "basin": basins,
                "time": times,
            },
        )

    def cache_forcing_xrdataset(self):
        """Save all daymet basin-forcing data in a netcdf file in the cache directory.

        """
        cache_npy_file = CACHE_DIR.joinpath("camels_daymet_forcing.npy")
        json_file = CACHE_DIR.joinpath("camels_daymet_forcing.json")
        if (not os.path.isfile(cache_npy_file)) or (not os.path.isfile(json_file)):
            self.cache_forcing_np_json()
        daymet_forcing = np.load(cache_npy_file)
        with open(json_file, "r") as fp:
            daymet_forcing_dict = json.load(
                fp, object_pairs_hook=collections.OrderedDict
            )
        import pint_xarray

        basins = daymet_forcing_dict["basin"]
        times = pd.date_range(
            daymet_forcing_dict["time"][0], periods=len(daymet_forcing_dict["time"])
        )
        variables = daymet_forcing_dict["variable"]
        # All units' names are from Pint https://github.com/hgrecco/pint/blob/master/pint/default_en.txt
        # final is PET's unit. PET comes from the model output of CAMELS-US
        units = ["s", "mm/day", "W/m^2", "mm", "°C", "°C", "Pa", "mm/day"]
        return xr.Dataset(
            data_vars={
                **{
                    variables[i]: (
                        ["basin", "time"],
                        daymet_forcing[:, :, i],
                        {"units": units[i]},
                    )
                    for i in range(len(variables))
                }
            },
            coords={
                "basin": basins,
                "time": times,
            },
            attrs={"forcing_type": "daymet"},
        )

    def cache_xrdataset(self):
        """Save all data in a netcdf file in the cache directory"""
        warnings.warn("Check you units of all variables")
        filename = "camels" + self.region.lower()
        filename_attributes = filename +"_attributes.nc"
        filename_timeseries = filename + "_timeseries.nc"
        ds_attr = self.cache_attributes_xrdataset()
        ds_attr.to_netcdf(CACHE_DIR.joinpath(filename_attributes))
        ds_streamflow = self.cache_streamflow_xrdataset()
        ds_forcing = self.cache_forcing_xrdataset()
        ds = xr.merge([ds_streamflow, ds_forcing])
        ds.to_netcdf(CACHE_DIR.joinpath(filename_timeseries))

    def read_ts_xrdataset(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs,
    ):
        if var_lst is None:
            return None
        filename = "camels" + self.region.lower()
        filename = filename + "_timeseries.nc"
        camels_tsnc = CACHE_DIR.joinpath(filename)
        if not os.path.isfile(camels_tsnc):
            self.cache_xrdataset()
        ts = xr.open_dataset(camels_tsnc)
        all_vars = ts.data_vars
        if any(var not in ts.variables for var in var_lst):
            raise ValueError(f"var_lst must all be in {all_vars}")
        return ts[var_lst].sel(basin=gage_id_lst, time=slice(t_range[0], t_range[1]))

    def read_attr_xrdataset(self, gage_id_lst=None, var_lst=None, **kwargs):
        if var_lst is None or len(var_lst) == 0:
            return None
        filename = "camels" + self.region.lower()
        filename = filename + "_attributes.nc"
        try:
            attr = xr.open_dataset(CACHE_DIR.joinpath(filename))
        except FileNotFoundError:
            attr = self.cache_attributes_xrdataset()
            attr.to_netcdf(CACHE_DIR.joinpath(filename))
        if "all_number" in list(kwargs.keys()) and kwargs["all_number"]:
            attr_num = map_string_vars(attr)
            return attr_num[var_lst].sel(basin=gage_id_lst)
        return attr[var_lst].sel(basin=gage_id_lst)

    @property
    def streamflow_unit(self):
        return "foot^3/s"


    def delete_variables_unit(self, variables):
        """
        delete the unit behind the variables name, e.g. 'prcp(mm/day)' -> 'prcp'

        Parameters
        ----------
        variables,

        Returns
        -------
        variables_list,
        """
        variables_list = []
        for name in variables:
            name_ = name.split("(")[0]
            variables_list.append(name_)
        return variables_list

    def unit_convert_mean_prcp(self, p_mean):
        """
        convert the mean precipitation uint to mm/d
        Parameters
        ----------
        p_mean

        Returns
        -------
        converted_data

        """
        if unit in ["mm/d", "mm/day"]:
            converted_data = p_mean
        elif unit in ["mm/h", "mm/hour"]:
            converted_data = p_mean / 24
        elif unit in ["mm/3h", "mm/3hour"]:
            converted_data = p_mean / 8
        elif unit in ["mm/8d", "mm/8day"]:
            converted_data = p_mean * 8
        elif unit in ["mm/y", "mm/year"]:
            converted_data = data * 365  # todo: whether or not to consider the leap year
        else:
            raise ValueError(
                "unit must be one of ['mm/d', 'mm/day', 'mm/h', 'mm/hour', 'mm/3h', 'mm/3hour', 'mm/8d', 'mm/8day', 'mm/y', 'mm/year']"
            )
        return converted_data

    def unit_convert_streamflow_m3tofoot3(self, Q):
        """
        convert the streamflow uint, m^3/s -> foot^3/s
        Parameters
        ----------
        Q
        m^3/s.
        Returns
        -------
        Q_foot
        foot^3/s.
        """
        Q_foot = flow * 35.314666721489
        return flow_foot

