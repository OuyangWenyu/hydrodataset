"""
Author: Wenyu Ouyang
Date: 2022-01-05 18:01:11
LastEditTime: 2024-09-10 19:15:33
LastEditors: Wenyu Ouyang
Description: Read Camels Series ("AUStralia", "BRazil", "ChiLe", "GreatBritain", "UnitedStates") datasets
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

        if self.region == "US":
            return self._set_data_source_camelsus_describe(camels_db)
        elif self.region == "AUS":
            return self._set_data_source_camelsaus_describe(camels_db)
        elif self.region == "BR":
            return self._set_data_source_camelsbr_describe(camels_db)
        elif self.region == "CL":
            return self._set_data_source_camelscl_describe(camels_db)
        elif self.region == "GB":
            return self._set_data_source_camelsgb_describe(camels_db)
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)

    def _set_data_source_camelsgb_describe(self, camels_db):
        # shp file of basins
        camels_shp_file = camels_db.joinpath(
            "8344e4f3-d2ea-44f5-8afa-86d2987543a9",
            "8344e4f3-d2ea-44f5-8afa-86d2987543a9",
            "data",
            "CAMELS_GB_catchment_boundaries",
            "CAMELS_GB_catchment_boundaries.shp",
        )
        # flow and forcing data are in a same file
        flow_dir = camels_db.joinpath(
            "8344e4f3-d2ea-44f5-8afa-86d2987543a9",
            "8344e4f3-d2ea-44f5-8afa-86d2987543a9",
            "data",
            "timeseries",
        )
        forcing_dir = flow_dir
        # attr
        attr_dir = camels_db.joinpath(
            "8344e4f3-d2ea-44f5-8afa-86d2987543a9",
            "8344e4f3-d2ea-44f5-8afa-86d2987543a9",
            "data",
        )
        gauge_id_file = attr_dir.joinpath("CAMELS_GB_hydrometry_attributes.csv")
        attr_key_lst = [
            "climatic",
            "humaninfluence",
            "hydrogeology",
            "hydrologic",
            "hydrometry",
            "landcover",
            "soil",
            "topographic",
        ]

        return collections.OrderedDict(
            CAMELS_DIR=camels_db,
            CAMELS_FLOW_DIR=flow_dir,
            CAMELS_FORCING_DIR=forcing_dir,
            CAMELS_ATTR_DIR=attr_dir,
            CAMELS_ATTR_KEY_LST=attr_key_lst,
            CAMELS_GAUGE_FILE=gauge_id_file,
            CAMELS_BASINS_SHP_FILE=camels_shp_file,
        )

    def _set_data_source_camelscl_describe(self, camels_db):
        # attr
        attr_dir = camels_db.joinpath("1_CAMELScl_attributes")
        attr_file = attr_dir.joinpath("1_CAMELScl_attributes.txt")
        # shp file of basins
        camels_shp_file = camels_db.joinpath(
            "CAMELScl_catchment_boundaries",
            "catchments_camels_cl_v1.3.shp",
        )
        # config of flow data
        flow_dir_m3s = camels_db.joinpath("2_CAMELScl_streamflow_m3s")
        flow_dir_mm = camels_db.joinpath("3_CAMELScl_streamflow_mm")

        # forcing
        forcing_dir_precip_cr2met = camels_db.joinpath("4_CAMELScl_precip_cr2met")
        forcing_dir_precip_chirps = camels_db.joinpath("5_CAMELScl_precip_chirps")
        forcing_dir_precip_mswep = camels_db.joinpath("6_CAMELScl_precip_mswep")
        forcing_dir_precip_tmpa = camels_db.joinpath("7_CAMELScl_precip_tmpa")
        forcing_dir_tmin_cr2met = camels_db.joinpath("8_CAMELScl_tmin_cr2met")
        forcing_dir_tmax_cr2met = camels_db.joinpath("9_CAMELScl_tmax_cr2met")
        forcing_dir_tmean_cr2met = camels_db.joinpath("10_CAMELScl_tmean_cr2met")
        forcing_dir_pet_8d_modis = camels_db.joinpath("11_CAMELScl_pet_8d_modis")
        forcing_dir_pet_hargreaves = camels_db.joinpath("12_CAMELScl_pet_hargreaves")
        forcing_dir_swe = camels_db.joinpath("13_CAMELScl_swe")
        return collections.OrderedDict(
            CAMELS_DIR=camels_db,
            CAMELS_FLOW_DIR=[flow_dir_m3s, flow_dir_mm],
            CAMELS_FORCING_DIR=[
                forcing_dir_precip_cr2met,
                forcing_dir_precip_chirps,
                forcing_dir_precip_mswep,
                forcing_dir_precip_tmpa,
                forcing_dir_tmin_cr2met,
                forcing_dir_tmax_cr2met,
                forcing_dir_tmean_cr2met,
                forcing_dir_pet_8d_modis,
                forcing_dir_pet_hargreaves,
                forcing_dir_swe,
            ],
            CAMELS_ATTR_DIR=attr_dir,
            CAMELS_GAUGE_FILE=attr_file,
            CAMELS_BASINS_SHP_FILE=camels_shp_file,
        )

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
            "14_CAMELS_BR_catchment_boundaries",
            "14_CAMELS_BR_catchment_boundaries",
            "camels_br_catchments.shp",
        )
        # config of flow data
        flow_dir_m3s = camels_db.joinpath(
            "02_CAMELS_BR_streamflow_m3s", "02_CAMELS_BR_streamflow_m3s"
        )
        flow_dir_mm_selected_catchments = camels_db.joinpath(
            "03_CAMELS_BR_streamflow_mm_selected_catchments",
            "03_CAMELS_BR_streamflow_mm_selected_catchments",
        )
        flow_dir_simulated = camels_db.joinpath(
            "04_CAMELS_BR_streamflow_simulated",
            "04_CAMELS_BR_streamflow_simulated",
        )

        # forcing
        forcing_dir_precipitation_chirps = camels_db.joinpath(
            "05_CAMELS_BR_precipitation_chirps",
            "05_CAMELS_BR_precipitation_chirps",
        )
        forcing_dir_precipitation_mswep = camels_db.joinpath(
            "06_CAMELS_BR_precipitation_mswep",
            "06_CAMELS_BR_precipitation_mswep",
        )
        forcing_dir_precipitation_cpc = camels_db.joinpath(
            "07_CAMELS_BR_precipitation_cpc",
            "07_CAMELS_BR_precipitation_cpc",
        )
        forcing_dir_evapotransp_gleam = camels_db.joinpath(
            "08_CAMELS_BR_evapotransp_gleam",
            "08_CAMELS_BR_evapotransp_gleam",
        )
        forcing_dir_evapotransp_mgb = camels_db.joinpath(
            "09_CAMELS_BR_evapotransp_mgb",
            "09_CAMELS_BR_evapotransp_mgb",
        )
        forcing_dir_potential_evapotransp_gleam = camels_db.joinpath(
            "10_CAMELS_BR_potential_evapotransp_gleam",
            "10_CAMELS_BR_potential_evapotransp_gleam",
        )
        forcing_dir_temperature_min_cpc = camels_db.joinpath(
            "11_CAMELS_BR_temperature_min_cpc",
            "11_CAMELS_BR_temperature_min_cpc",
        )
        forcing_dir_temperature_mean_cpc = camels_db.joinpath(
            "12_CAMELS_BR_temperature_mean_cpc",
            "12_CAMELS_BR_temperature_mean_cpc",
        )
        forcing_dir_temperature_max_cpc = camels_db.joinpath(
            "13_CAMELS_BR_temperature_max_cpc",
            "13_CAMELS_BR_temperature_max_cpc",
        )
        return collections.OrderedDict(
            CAMELS_DIR=camels_db,
            CAMELS_FLOW_DIR=[
                flow_dir_m3s,
                flow_dir_mm_selected_catchments,
                flow_dir_simulated,
            ],
            CAMELS_FORCING_DIR=[
                forcing_dir_precipitation_chirps,
                forcing_dir_precipitation_mswep,
                forcing_dir_precipitation_cpc,
                forcing_dir_evapotransp_gleam,
                forcing_dir_evapotransp_mgb,
                forcing_dir_potential_evapotransp_gleam,
                forcing_dir_temperature_min_cpc,
                forcing_dir_temperature_mean_cpc,
                forcing_dir_temperature_max_cpc,
            ],
            CAMELS_ATTR_DIR=attr_dir,
            CAMELS_ATTR_KEY_LST=attr_key_lst,
            CAMELS_GAUGE_FILE=gauge_id_file,
            CAMELS_BASINS_SHP_FILE=camels_shp_file,
        )

    def _set_data_source_camelsaus_describe(self, camels_db):
        # id and name
        gauge_id_file = camels_db.joinpath(
            "01_id_name_metadata",
            "01_id_name_metadata",
            "id_name_metadata.csv",
        )
        # shp file of basins
        camels_shp_file = camels_db.joinpath(
            "02_location_boundary_area",
            "02_location_boundary_area",
            "shp",
            "CAMELS_AUS_BasinOutlets_adopted.shp",
        )
        # config of flow data
        flow_dir = camels_db.joinpath("03_streamflow", "03_streamflow")
        # attr
        attr_dir = camels_db.joinpath("04_attributes", "04_attributes")
        # forcing
        forcing_dir = camels_db.joinpath("05_hydrometeorology", "05_hydrometeorology")

        return collections.OrderedDict(
            CAMELS_DIR=camels_db,
            CAMELS_FLOW_DIR=flow_dir,
            CAMELS_FORCING_DIR=forcing_dir,
            CAMELS_ATTR_DIR=attr_dir,
            CAMELS_GAUGE_FILE=gauge_id_file,
            CAMELS_BASINS_SHP_FILE=camels_shp_file,
        )

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
        if self.region == "US":
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
        else:
            warnings.warn("We only provide downloading methods for CAMELS-US now")
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
        if self.region == "US":
            data = pd.read_csv(
                camels_file, sep=";", dtype={"gauge_id": str, "huc_02": str}
            )
        elif self.region == "AUS":
            data = pd.read_csv(camels_file, sep=",", dtype={"station_id": str})
        elif self.region == "BR":
            data = pd.read_csv(camels_file, sep="\s+", dtype={"gauge_id": str})
        elif self.region == "CL":
            data = pd.read_csv(camels_file, sep="\t", index_col=0)
        elif self.region == "GB":
            data = pd.read_csv(camels_file, sep=",", dtype={"gauge_id": str})
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)
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
        if self.region == "US":
            return self._get_constant_cols_some(data_folder, "camels_", ".txt", ";")
        elif self.region == "AUS":
            attr_all_file = os.path.join(
                self.data_source_description["CAMELS_DIR"],
                "CAMELS_AUS_Attributes-Indices_MasterTable.csv",
            )
            camels_aus_attr_indices_data = pd.read_csv(attr_all_file, sep=",")
            # exclude station id
            return camels_aus_attr_indices_data.columns.values[1:]
        elif self.region == "BR":
            return self._get_constant_cols_some(
                data_folder, "camels_br_", ".txt", "\s+"
            )
        elif self.region == "CL":
            camels_cl_attr_data = self.sites
            # exclude station id
            return camels_cl_attr_data.index.values
        elif self.region == "GB":
            return self._get_constant_cols_some(
                data_folder, "CAMELS_GB_", "_attributes.csv", ","
            )
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)

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
        if self.region == "US":
            # PET is from model_output file in CAMELS-US
            return np.array(
                ["dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp", "PET"]
            )
        elif self.region == "AUS":
            forcing_types = []
            for root, dirs, files in os.walk(
                self.data_source_description["CAMELS_FORCING_DIR"]
            ):
                if root == self.data_source_description["CAMELS_FORCING_DIR"]:
                    continue
                forcing_types.extend(
                    file[:-4] for file in files if file != "ClimaticIndices.csv"
                )
            return np.array(forcing_types)
        elif self.region == "BR":
            return np.array(
                [
                    str(forcing_dir).split(os.sep)[-1][13:]
                    for forcing_dir in self.data_source_description[
                        "CAMELS_FORCING_DIR"
                    ]
                ]
            )
        elif self.region == "CL":
            return np.array(
                [
                    "_".join(str(forcing_dir).split(os.sep)[-1].split("_")[2:])
                    for forcing_dir in self.data_source_description[
                        "CAMELS_FORCING_DIR"
                    ]
                ]
            )
        elif self.region == "GB":
            return np.array(
                [
                    "precipitation",
                    "pet",
                    "temperature",
                    "peti",
                    "humidity",
                    "shortwave_rad",
                    "longwave_rad",
                    "windspeed",
                ]
            )
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)

    def get_target_cols(self) -> np.ndarray:
        """
        For CAMELS, the target vars are streamflows

        Returns
        -------
        np.array
            streamflow types
        """
        if self.region == "US":
            return np.array(["usgsFlow", "ET"])
        elif self.region == "AUS":
            # QualityCodes are not streamflow data.
            # MLd means "1 Megaliters Per Day"; 1 MLd = 0.011574074074074 cubic-meters-per-second
            # mmd means "mm/day"
            return np.array(
                [
                    "streamflow_MLd",
                    "streamflow_MLd_inclInfilled",
                    "streamflow_mmd",
                    "streamflow_QualityCodes",
                ]
            )
        elif self.region == "BR":
            return np.array(
                [
                    str(flow_dir).split(os.sep)[-1][13:]
                    for flow_dir in self.data_source_description["CAMELS_FLOW_DIR"]
                ]
            )
        elif self.region == "CL":
            return np.array(
                [
                    str(flow_dir).split(os.sep)[-1][11:]
                    for flow_dir in self.data_source_description["CAMELS_FLOW_DIR"]
                ]
            )
        elif self.region == "GB":
            return np.array(["discharge_spec", "discharge_vol"])
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)

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
        if self.region in ["BR", "GB", "US"]:
            return self.sites["gauge_id"].values
        elif self.region == "AUS":
            return self.sites["station_id"].values
        elif self.region == "CL":
            station_ids = self.sites.columns.values
            # for 7-digit id, replace the space with 0 to get a 8-digit id
            cl_station_ids = [
                station_id.split(" ")[-1].zfill(8) for station_id in station_ids
            ]
            return np.array(cl_station_ids)
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)

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

    def read_br_gage_flow(self, gage_id, t_range, flow_type):
        """
        Read gage's streamflow from CAMELS-BR

        Parameters
        ----------
        gage_id
            the station id
        t_range
            the time range, for example, ["1990-01-01", "2000-01-01"]
        flow_type
            "streamflow_m3s" or "streamflow_mm_selected_catchments" or "streamflow_simulated"

        Returns
        -------
        np.array
            streamflow data of one station for a given time range
        """
        logging.debug("reading %s streamflow data", gage_id)
        dir_ = [
            str(flow_dir)
            for flow_dir in self.data_source_description["CAMELS_FLOW_DIR"]
            if flow_type in str(flow_dir)
        ][0]
        if flow_type == "streamflow_mm_selected_catchments":
            flow_type = "streamflow_mm"
        elif flow_type == "streamflow_simulated":
            flow_type = "simulated_streamflow"
        gage_file = os.path.join(dir_, gage_id + "_" + flow_type + ".txt")
        data_temp = pd.read_csv(gage_file, sep=r"\s+")
        obs = data_temp.iloc[:, 3].values
        obs[obs < 0] = np.nan
        df_date = data_temp[["year", "month", "day"]]
        date = pd.to_datetime(df_date).values.astype("datetime64[D]")
        return time_intersect_dynamic_data(obs, date, t_range)

    def read_gb_gage_flow_forcing(self, gage_id, t_range, var_type):
        """
        Read gage's streamflow or forcing from CAMELS-GB

        Parameters
        ----------
        gage_id
            the station id
        t_range
            the time range, for example, ["1990-01-01", "2000-01-01"]
        var_type
            flow type: "discharge_spec" or "discharge_vol"
            forcing type: "precipitation", "pet", "temperature", "peti", "humidity", "shortwave_rad", "longwave_rad",
                          "windspeed"

        Returns
        -------
        np.array
            streamflow or forcing data of one station for a given time range
        """
        logging.debug("reading %s streamflow data", gage_id)
        gage_file = os.path.join(
            self.data_source_description["CAMELS_FLOW_DIR"],
            "CAMELS_GB_hydromet_timeseries_" + gage_id + "_19701001-20150930.csv",
        )
        data_temp = pd.read_csv(gage_file, sep=",")
        obs = data_temp[var_type].values
        if var_type in ["discharge_spec", "discharge_vol"]:
            obs[obs < 0] = np.nan
        date = pd.to_datetime(data_temp["date"]).values.astype("datetime64[D]")
        return time_intersect_dynamic_data(obs, date, t_range)

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
        t_range_list = hydro_time.t_range_days(t_range)
        nt = t_range_list.shape[0]
        y = np.full([len(gage_id_lst), nt, nf], np.nan)
        if self.region == "US":
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
        elif self.region == "AUS":
            for k in tqdm(
                range(len(target_cols)), desc="Read streamflow data of CAMELS-AUS"
            ):
                flow_data = pd.read_csv(
                    os.path.join(
                        self.data_source_description["CAMELS_FLOW_DIR"],
                        target_cols[k] + ".csv",
                    )
                )
                df_date = flow_data[["year", "month", "day"]]
                date = pd.to_datetime(df_date).values.astype("datetime64[D]")
                [c, ind1, ind2] = np.intersect1d(
                    date, t_range_list, return_indices=True
                )
                chosen_data = flow_data[gage_id_lst].values[ind1, :]
                chosen_data[chosen_data < 0] = np.nan
                y[:, ind2, k] = chosen_data.T
        elif self.region == "BR":
            for j in tqdm(
                range(len(target_cols)), desc="Read streamflow data of CAMELS-BR"
            ):
                for k in tqdm(range(len(gage_id_lst))):
                    data_obs = self.read_br_gage_flow(
                        gage_id_lst[k], t_range, target_cols[j]
                    )
                    y[k, :, j] = data_obs
        elif self.region == "CL":
            for k in tqdm(
                range(len(target_cols)), desc="Read streamflow data of CAMELS-CL"
            ):
                if target_cols[k] == "streamflow_m3s":
                    flow_data = pd.read_csv(
                        os.path.join(
                            self.data_source_description["CAMELS_FLOW_DIR"][0],
                            "2_CAMELScl_streamflow_m3s.txt",
                        ),
                        sep="\t",
                        index_col=0,
                    )
                elif target_cols[k] == "streamflow_mm":
                    flow_data = pd.read_csv(
                        os.path.join(
                            self.data_source_description["CAMELS_FLOW_DIR"][1],
                            "3_CAMELScl_streamflow_mm.txt",
                        ),
                        sep="\t",
                        index_col=0,
                    )
                else:
                    raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)
                date = pd.to_datetime(flow_data.index.values).values.astype(
                    "datetime64[D]"
                )
                [c, ind1, ind2] = np.intersect1d(
                    date, t_range_list, return_indices=True
                )
                station_ids = [id_.zfill(8) for id_ in flow_data.columns.values]
                assert all(x < y for x, y in zip(station_ids, station_ids[1:]))
                ind3 = [station_ids.index(tmp) for tmp in gage_id_lst]
                # to guarantee the sequence is not changed we don't use np.intersect1d
                chosen_data = flow_data.iloc[ind1, ind3].replace(
                    "\s+", np.nan, regex=True
                )
                chosen_data = chosen_data.astype(float)
                chosen_data[chosen_data < 0] = np.nan
                y[:, ind2, k] = chosen_data.values.T
        elif self.region == "GB":
            for j in tqdm(
                range(len(target_cols)), desc="Read streamflow data of CAMELS-GB"
            ):
                for k in tqdm(range(len(gage_id_lst))):
                    data_obs = self.read_gb_gage_flow_forcing(
                        gage_id_lst[k], t_range, target_cols[j]
                    )
                    y[k, :, j] = data_obs
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)
        # Keep unit of streamflow unified: we use ft3/s here
        # unit of flow in AUS is MegaLiter/day -> ft3/s
        if self.region == "AUS":
            y = y / 86.4 * 35.314666721489
        elif self.region != "US":
            # other units are m3/s -> ft3/s
            y = y * 35.314666721489
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

    def read_br_basin_forcing(self, gage_id, t_range, var_type) -> np.array:
        """
        Read one forcing data for a basin in CAMELS_BR

        Parameters
        ----------
        gage_id
            basin id
        t_range
            the time range, for example, ["1995-01-01", "2005-01-01"]
        var_type
            the forcing variable type

        Returns
        -------
        np.array
            one type forcing data of a basin in a given time range
        """
        dir_ = [
            str(_dir)
            for _dir in self.data_source_description["CAMELS_FORCING_DIR"]
            if var_type in str(_dir)
        ][0]
        if var_type in [
            "temperature_min_cpc",
            "temperature_mean_cpc",
            "temperature_max_cpc",
        ]:
            var_type = var_type[:-4]
        gage_file = os.path.join(dir_, gage_id + "_" + var_type + ".txt")
        data_temp = pd.read_csv(gage_file, sep=r"\s+")
        obs = data_temp.iloc[:, 3].values
        df_date = data_temp[["year", "month", "day"]]
        date = pd.to_datetime(df_date).values.astype("datetime64[D]")
        return time_intersect_dynamic_data(obs, date, t_range)

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
        if self.region == "US":
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
        elif self.region == "AUS":
            for k in tqdm(range(len(var_lst)), desc="Read forcing data of CAMELS-AUS"):
                if "precipitation_" in var_lst[k]:
                    forcing_dir = os.path.join(
                        self.data_source_description["CAMELS_FORCING_DIR"],
                        "01_precipitation_timeseries",
                    )
                elif "et_" in var_lst[k] or "evap_" in var_lst[k]:
                    forcing_dir = os.path.join(
                        self.data_source_description["CAMELS_FORCING_DIR"],
                        "02_EvaporativeDemand_timeseries",
                    )
                elif "_AWAP" in var_lst[k]:
                    forcing_dir = os.path.join(
                        self.data_source_description["CAMELS_FORCING_DIR"],
                        "03_Other",
                        "AWAP",
                    )
                elif "_SILO" in var_lst[k]:
                    forcing_dir = os.path.join(
                        self.data_source_description["CAMELS_FORCING_DIR"],
                        "03_Other",
                        "SILO",
                    )
                else:
                    raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)
                forcing_data = pd.read_csv(
                    os.path.join(forcing_dir, var_lst[k] + ".csv")
                )
                df_date = forcing_data[["year", "month", "day"]]
                date = pd.to_datetime(df_date).values.astype("datetime64[D]")
                [c, ind1, ind2] = np.intersect1d(
                    date, t_range_list, return_indices=True
                )
                chosen_data = forcing_data[gage_id_lst].values[ind1, :]
                x[:, ind2, k] = chosen_data.T
        elif self.region == "BR":
            for j in tqdm(range(len(var_lst)), desc="Read forcing data of CAMELS-BR"):
                for k in tqdm(range(len(gage_id_lst))):
                    data_obs = self.read_br_basin_forcing(
                        gage_id_lst[k], t_range, var_lst[j]
                    )
                    x[k, :, j] = data_obs
        elif self.region == "CL":
            for k in tqdm(range(len(var_lst)), desc="Read forcing data of CAMELS-CL"):
                for tmp in os.listdir(self.data_source_description["CAMELS_DIR"]):
                    if fnmatch.fnmatch(tmp, "*" + var_lst[k]):
                        tmp_ = os.path.join(
                            self.data_source_description["CAMELS_DIR"], tmp
                        )
                        if os.path.isdir(tmp_):
                            forcing_file = os.path.join(tmp_, os.listdir(tmp_)[0])
                forcing_data = pd.read_csv(forcing_file, sep="\t", index_col=0)
                date = pd.to_datetime(forcing_data.index.values).values.astype(
                    "datetime64[D]"
                )
                [c, ind1, ind2] = np.intersect1d(
                    date, t_range_list, return_indices=True
                )
                station_ids = [id_.zfill(8) for id_ in forcing_data.columns.values]
                assert all(x < y for x, y in zip(station_ids, station_ids[1:]))
                ind3 = [station_ids.index(tmp) for tmp in gage_id_lst]
                # to guarantee the sequence is not changed we don't use np.intersect1d
                chosen_data = forcing_data.iloc[ind1, ind3].replace(
                    "\s+", np.nan, regex=True
                )
                x[:, ind2, k] = chosen_data.values.T
        elif self.region == "GB":
            for j in tqdm(range(len(var_lst)), desc="Read forcing data of CAMELS-GB"):
                for k in tqdm(range(len(gage_id_lst))):
                    data_forcing = self.read_gb_gage_flow_forcing(
                        gage_id_lst[k], t_range, var_lst[j]
                    )
                    x[k, :, j] = data_forcing
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)
        return x

    def read_attr_all(self):
        data_folder = self.data_source_description["CAMELS_ATTR_DIR"]
        key_lst = self.data_source_description["CAMELS_ATTR_KEY_LST"]
        f_dict = {}
        var_dict = {}
        var_lst = []
        out_lst = []
        gage_dict = self.sites
        if self.region == "US":
            camels_str = "camels_"
            sep_ = ";"
        elif self.region == "BR":
            camels_str = "camels_br_"
            sep_ = "\s+"
        elif self.region == "GB":
            camels_str = "CAMELS_GB_"
            sep_ = ","
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)
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

    def read_attr_all_in_one_file(self):
        """
        Read all attr data in CAMELS_AUS or CAMELS_CL

        Returns
        -------
        np.array
            all attr data in CAMELS_AUS or CAMELS_CL
        """
        if self.region == "AUS":
            attr_all_file = os.path.join(
                self.data_source_description["CAMELS_DIR"],
                "CAMELS_AUS_Attributes-Indices_MasterTable.csv",
            )
            all_attr = pd.read_csv(attr_all_file, sep=",")
        elif self.region == "CL":
            attr_all_file = os.path.join(
                self.data_source_description["CAMELS_ATTR_DIR"],
                "1_CAMELScl_attributes.txt",
            )
            all_attr_tmp = pd.read_csv(attr_all_file, sep="\t", index_col=0)
            all_attr = pd.DataFrame(
                all_attr_tmp.values.T,
                index=all_attr_tmp.columns,
                columns=all_attr_tmp.index,
            )
            # some none str attributes are treated as str, we need to trans them to float
            all_cols = all_attr.columns
            for col in all_cols:
                try:
                    all_attr[col] = all_attr[col].astype(float)
                except Exception:
                    continue
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)
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
            if true, return var_dict and f_dict for CAMELS_US
        Returns
        -------
        Union[tuple, np.array]
            if attr var type is str, return factorized data.
            When we need to know what a factorized value represents, we need return a tuple;
            otherwise just return an array
        """
        if self.region in ["BR", "GB", "US"]:
            attr_all, var_lst_all, var_dict, f_dict = self.read_attr_all()
        elif self.region in ["AUS", "CL"]:
            attr_all, var_lst_all, var_dict, f_dict = self.read_attr_all_in_one_file()
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)
        ind_var = [var_lst_all.index(var) for var in var_lst]
        id_lst_all = self.read_object_ids()
        # Notice the sequence of station ids ! Some id_lst_all are not sorted, so don't use np.intersect1d
        ind_grid = [id_lst_all.tolist().index(tmp) for tmp in gage_id_lst]
        temp = attr_all[ind_grid, :]
        out = temp[:, ind_var]
        return (out, var_dict, f_dict) if is_return_dict else out

    def read_area(self, gage_id_lst) -> np.ndarray:
        if self.region == "US":
            return self.read_attr_xrdataset(gage_id_lst, ["area_gages2"])
        elif self.region == "AUS":
            return self.read_constant_cols(
                gage_id_lst, ["catchment_area"], is_return_dict=False
            )
        elif self.region in ["BR", "CL", "GB"]:
            return self.read_constant_cols(gage_id_lst, ["area"], is_return_dict=False)
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)

    def read_mean_prcp(self, gage_id_lst) -> np.ndarray:
        if self.region in ["US", "AUS", "BR", "GB"]:
            if self.region == "US":
                return self.read_attr_xrdataset(gage_id_lst, ["p_mean"])
            return self.read_constant_cols(
                gage_id_lst, ["p_mean"], is_return_dict=False
            )
        elif self.region == "CL":
            # there are different p_mean values for different forcings, here we chose p_mean_cr2met now
            return self.read_constant_cols(
                gage_id_lst, ["p_mean_cr2met"], is_return_dict=False
            )
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)

    def cache_forcing_np_json(self):
        """
        Save all daymet basin-forcing data in a numpy array file in the cache directory.

        Because it takes much time to read data from txt files,
        it is a good way to cache data as a numpy file to speed up the reading.
        In addition, we need a document to explain the meaning of all dimensions.

        TODO: now only support CAMELS-US
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

        TODO: now only support CAMELS-US
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
        TODO: now only support CAMELS-US

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

        TODO: ONLY SUPPORT CAMELS-US now
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

        TODO: ONLY SUPPORT CAMELS-US now
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
        ds_attr = self.cache_attributes_xrdataset()
        ds_attr.to_netcdf(CACHE_DIR.joinpath("camelsus_attributes.nc"))
        ds_streamflow = self.cache_streamflow_xrdataset()
        ds_forcing = self.cache_forcing_xrdataset()
        ds = xr.merge([ds_streamflow, ds_forcing])
        ds.to_netcdf(CACHE_DIR.joinpath("camelsus_timeseries.nc"))

    def read_ts_xrdataset(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs,
    ):
        if var_lst is None:
            return None
        camels_tsnc = CACHE_DIR.joinpath("camelsus_timeseries.nc")
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
        try:
            attr = xr.open_dataset(CACHE_DIR.joinpath("camelsus_attributes.nc"))
        except FileNotFoundError:
            attr = self.cache_attributes_xrdataset()
            attr.to_netcdf(CACHE_DIR.joinpath("camelsus_attributes.nc"))
        if "all_number" in list(kwargs.keys()) and kwargs["all_number"]:
            attr_num = map_string_vars(attr)
            return attr_num[var_lst].sel(basin=gage_id_lst)
        return attr[var_lst].sel(basin=gage_id_lst)

    @property
    def streamflow_unit(self):
        return "foot^3/s"
