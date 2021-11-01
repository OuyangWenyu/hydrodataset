import collections
import fnmatch
import os
from typing import Union

import pandas as pd
import numpy as np
from pandas.core.dtypes.common import is_string_dtype, is_numeric_dtype

from hydrobench.data.data_base import DataSourceBase
from hydrobench.data.stat import cal_fdc
from hydrobench.utils import hydro_utils
from hydrobench.utils.hydro_utils import download_one_zip, unzip_nested_zip

CAMELS_NO_DATASET_ERROR_LOG = "We cannot read this dataset now. Please check if you choose the correct dataset:\n" \
                              " [\"AUS\", \"BR\", \"CL\", \"GB\", \"US\", \"YR\"]"


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
    t_lst = hydro_utils.t_range_days(t_range)
    nt = t_lst.shape[0]
    if len(obs) != nt:
        out = np.full([nt], np.nan)
        [c, ind1, ind2] = np.intersect1d(date, t_lst, return_indices=True)
        out[ind2] = obs[ind1]
    else:
        out = obs
    return out


class Camels(DataSourceBase):
    def __init__(self, data_path, download=False, region: str = "US"):
        """
        Initialization for CAMELS series dataset

        Parameters
        ----------
        data_path
            where we put the dataset
        download
            if true, download
        region
            the default is CAMELS(-US), since it's the first CAMELS dataset.
            Others now include: AUS, BR, CL, GB, YR
        """
        super().__init__(data_path)
        region_lst = ["AUS", "BR", "CL", "GB", "US", "YR"]
        assert region in region_lst
        self.region = region
        self.data_source_description = self.set_data_source_describe()
        if download:
            self.download_data_source()
        self.camels_sites = self.read_site_info()

    def get_name(self):
        return "CAMELS"

    def set_data_source_describe(self) -> collections.OrderedDict:
        """
        Introduce the files in the dataset and list their location in the file system

        Returns
        -------
        collections.OrderedDict
            the description for a CAMELS dataset
        """
        camels_db = self.data_source_dir
        if self.region == "US":
            # shp file of basins
            camels_shp_file = os.path.join(camels_db, "basin_set_full_res", "HCDN_nhru_final_671.shp")
            # config of flow data
            flow_dir = os.path.join(camels_db, "basin_timeseries_v1p2_metForcing_obsFlow", "basin_dataset_public_v1p2",
                                    "usgs_streamflow")
            # forcing
            forcing_dir = os.path.join(camels_db, "basin_timeseries_v1p2_metForcing_obsFlow",
                                       "basin_dataset_public_v1p2",
                                       "basin_mean_forcing")
            forcing_types = ["daymet", "maurer", "nldas"]
            # attr
            attr_dir = os.path.join(camels_db, "camels_attributes_v2.0", "camels_attributes_v2.0")
            gauge_id_file = os.path.join(attr_dir, 'camels_name.txt')
            attr_key_lst = ['topo', 'clim', 'hydro', 'vege', 'soil', 'geol']
            download_url_lst = [
                "https://ral.ucar.edu/sites/default/files/public/product-tool/camels-catchment-attributes-and-meteorology-for-large-sample-studies-dataset-downloads/camels_attributes_v2.0.zip",
                "https://ral.ucar.edu/sites/default/files/public/product-tool/camels-catchment-attributes-and-meteorology-for-large-sample-studies-dataset-downloads/basin_set_full_res.zip",
                "https://ral.ucar.edu/sites/default/files/public/product-tool/camels-catchment-attributes-and-meteorology-for-large-sample-studies-dataset-downloads/basin_timeseries_v1p2_metForcing_obsFlow.zip"]

            return collections.OrderedDict(CAMELS_DIR=camels_db, CAMELS_FLOW_DIR=flow_dir,
                                           CAMELS_FORCING_DIR=forcing_dir, CAMELS_FORCING_TYPE=forcing_types,
                                           CAMELS_ATTR_DIR=attr_dir, CAMELS_ATTR_KEY_LST=attr_key_lst,
                                           CAMELS_GAUGE_FILE=gauge_id_file,
                                           CAMELS_BASINS_SHP_FILE=camels_shp_file,
                                           CAMELS_DOWNLOAD_URL_LST=download_url_lst)
        elif self.region == "AUS":
            # id and name
            gauge_id_file = os.path.join(camels_db, "01_id_name_metadata", "01_id_name_metadata",
                                         "id_name_metadata.csv")
            # shp file of basins
            camels_shp_file = os.path.join(camels_db, "02_location_boundary_area", "02_location_boundary_area", "shp",
                                           "CAMELS_AUS_BasinOutlets_adopted.shp")
            # config of flow data
            flow_dir = os.path.join(camels_db, "03_streamflow", "03_streamflow")
            # attr
            attr_dir = os.path.join(camels_db, "04_attributes", "04_attributes")
            # forcing
            forcing_dir = os.path.join(camels_db, "05_hydrometeorology", "05_hydrometeorology")

            return collections.OrderedDict(CAMELS_DIR=camels_db, CAMELS_FLOW_DIR=flow_dir,
                                           CAMELS_FORCING_DIR=forcing_dir,
                                           CAMELS_ATTR_DIR=attr_dir, CAMELS_GAUGE_FILE=gauge_id_file,
                                           CAMELS_BASINS_SHP_FILE=camels_shp_file)
        elif self.region == "BR":
            # attr
            attr_dir = os.path.join(camels_db, "01_CAMELS_BR_attributes", "01_CAMELS_BR_attributes")
            # we don't need the location attr file
            attr_key_lst = ['climate', 'geology', 'human_intervention', 'hydrology', 'land_cover',
                            'quality_check', 'soil', 'topography']
            # id and name, there are two types stations in CAMELS_BR, and we only chose the 897-stations version
            gauge_id_file = os.path.join(attr_dir, "camels_br_topography.txt")
            # shp file of basins
            camels_shp_file = os.path.join(camels_db, "14_CAMELS_BR_catchment_boundaries",
                                           "14_CAMELS_BR_catchment_boundaries", "camels_br_catchments.shp")
            # config of flow data
            flow_dir_m3s = os.path.join(camels_db, "02_CAMELS_BR_streamflow_m3s", "02_CAMELS_BR_streamflow_m3s")
            flow_dir_mm_selected_catchments = os.path.join(camels_db, "03_CAMELS_BR_streamflow_mm_selected_catchments",
                                                           "03_CAMELS_BR_streamflow_mm_selected_catchments")
            flow_dir_simulated = os.path.join(camels_db, "04_CAMELS_BR_streamflow_simulated",
                                              "04_CAMELS_BR_streamflow_simulated")

            # forcing
            forcing_dir_precipitation_chirps = os.path.join(camels_db, "05_CAMELS_BR_precipitation_chirps",
                                                            "05_CAMELS_BR_precipitation_chirps")
            forcing_dir_precipitation_mswep = os.path.join(camels_db, "06_CAMELS_BR_precipitation_mswep",
                                                           "06_CAMELS_BR_precipitation_mswep")
            forcing_dir_precipitation_cpc = os.path.join(camels_db, "07_CAMELS_BR_precipitation_cpc",
                                                         "07_CAMELS_BR_precipitation_cpc")
            forcing_dir_evapotransp_gleam = os.path.join(camels_db, "08_CAMELS_BR_evapotransp_gleam",
                                                         "08_CAMELS_BR_evapotransp_gleam")
            forcing_dir_evapotransp_mgb = os.path.join(camels_db, "09_CAMELS_BR_evapotransp_mgb",
                                                       "09_CAMELS_BR_evapotransp_mgb")
            forcing_dir_potential_evapotransp_gleam = os.path.join(camels_db,
                                                                   "10_CAMELS_BR_potential_evapotransp_gleam",
                                                                   "10_CAMELS_BR_potential_evapotransp_gleam")
            forcing_dir_temperature_min_cpc = os.path.join(camels_db, "11_CAMELS_BR_temperature_min_cpc",
                                                           "11_CAMELS_BR_temperature_min_cpc")
            forcing_dir_temperature_mean_cpc = os.path.join(camels_db, "12_CAMELS_BR_temperature_mean_cpc",
                                                            "12_CAMELS_BR_temperature_mean_cpc")
            forcing_dir_temperature_max_cpc = os.path.join(camels_db, "13_CAMELS_BR_temperature_max_cpc",
                                                           "13_CAMELS_BR_temperature_max_cpc")
            return collections.OrderedDict(CAMELS_DIR=camels_db,
                                           CAMELS_FLOW_DIR=[flow_dir_m3s, flow_dir_mm_selected_catchments,
                                                            flow_dir_simulated],
                                           CAMELS_FORCING_DIR=[forcing_dir_precipitation_chirps,
                                                               forcing_dir_precipitation_mswep,
                                                               forcing_dir_precipitation_cpc,
                                                               forcing_dir_evapotransp_gleam,
                                                               forcing_dir_evapotransp_mgb,
                                                               forcing_dir_potential_evapotransp_gleam,
                                                               forcing_dir_temperature_min_cpc,
                                                               forcing_dir_temperature_mean_cpc,
                                                               forcing_dir_temperature_max_cpc],
                                           CAMELS_ATTR_DIR=attr_dir, CAMELS_ATTR_KEY_LST=attr_key_lst,
                                           CAMELS_GAUGE_FILE=gauge_id_file,
                                           CAMELS_BASINS_SHP_FILE=camels_shp_file)
        elif self.region == "CL":
            # attr
            attr_dir = os.path.join(camels_db, "1_CAMELScl_attributes")
            attr_file = os.path.join(attr_dir, "1_CAMELScl_attributes.txt")
            # shp file of basins
            camels_shp_file = os.path.join(camels_db, "CAMELScl_catchment_boundaries", "catchments_camels_cl_v1.3.shp")
            # config of flow data
            flow_dir_m3s = os.path.join(camels_db, "2_CAMELScl_streamflow_m3s")
            flow_dir_mm = os.path.join(camels_db, "3_CAMELScl_streamflow_mm")

            # forcing
            forcing_dir_precip_cr2met = os.path.join(camels_db, "4_CAMELScl_precip_cr2met")
            forcing_dir_precip_chirps = os.path.join(camels_db, "5_CAMELScl_precip_chirps")
            forcing_dir_precip_mswep = os.path.join(camels_db, "6_CAMELScl_precip_mswep")
            forcing_dir_precip_tmpa = os.path.join(camels_db, "7_CAMELScl_precip_tmpa")
            forcing_dir_tmin_cr2met = os.path.join(camels_db, "8_CAMELScl_tmin_cr2met")
            forcing_dir_tmax_cr2met = os.path.join(camels_db, "9_CAMELScl_tmax_cr2met")
            forcing_dir_tmean_cr2met = os.path.join(camels_db, "10_CAMELScl_tmean_cr2met")
            forcing_dir_pet_8d_modis = os.path.join(camels_db, "11_CAMELScl_pet_8d_modis")
            forcing_dir_pet_hargreaves = os.path.join(camels_db, "12_CAMELScl_pet_hargreaves", )
            forcing_dir_swe = os.path.join(camels_db, "13_CAMELScl_swe")
            return collections.OrderedDict(CAMELS_DIR=camels_db,
                                           CAMELS_FLOW_DIR=[flow_dir_m3s, flow_dir_mm],
                                           CAMELS_FORCING_DIR=[forcing_dir_precip_cr2met, forcing_dir_precip_chirps,
                                                               forcing_dir_precip_mswep, forcing_dir_precip_tmpa,
                                                               forcing_dir_tmin_cr2met, forcing_dir_tmax_cr2met,
                                                               forcing_dir_tmean_cr2met, forcing_dir_pet_8d_modis,
                                                               forcing_dir_pet_hargreaves, forcing_dir_swe],
                                           CAMELS_ATTR_DIR=attr_dir,
                                           CAMELS_GAUGE_FILE=attr_file,
                                           CAMELS_BASINS_SHP_FILE=camels_shp_file)
        elif self.region == "GB":
            # shp file of basins
            camels_shp_file = os.path.join(camels_db, "8344e4f3-d2ea-44f5-8afa-86d2987543a9",
                                           "8344e4f3-d2ea-44f5-8afa-86d2987543a9", "data",
                                           "CAMELS_GB_catchment_boundaries", "CAMELS_GB_catchment_boundaries.shp")
            # flow and forcing data are in a same file
            flow_dir = os.path.join(camels_db, "8344e4f3-d2ea-44f5-8afa-86d2987543a9",
                                    "8344e4f3-d2ea-44f5-8afa-86d2987543a9", "data", "timeseries")
            forcing_dir = flow_dir
            # attr
            attr_dir = os.path.join(camels_db, "8344e4f3-d2ea-44f5-8afa-86d2987543a9",
                                    "8344e4f3-d2ea-44f5-8afa-86d2987543a9", "data")
            gauge_id_file = os.path.join(attr_dir, 'CAMELS_GB_hydrometry_attributes.csv')
            attr_key_lst = ["climatic", "humaninfluence", "hydrogeology", "hydrologic", "hydrometry", "landcover",
                            "soil", "topographic"]

            return collections.OrderedDict(CAMELS_DIR=camels_db, CAMELS_FLOW_DIR=flow_dir,
                                           CAMELS_FORCING_DIR=forcing_dir, CAMELS_ATTR_DIR=attr_dir,
                                           CAMELS_ATTR_KEY_LST=attr_key_lst, CAMELS_GAUGE_FILE=gauge_id_file,
                                           CAMELS_BASINS_SHP_FILE=camels_shp_file)
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)

    def download_data_source(self) -> None:
        """
        Download CAMELS dataset.

        Now we only support CAMELS-US's downloading.
        For others, please download it manually and put all files of a CAMELS dataset in one directory.
        For example, all files of CAMELS_AUS should be put in "camels_aus" directory

        Returns
        -------
        None
        """
        camels_config = self.data_source_description
        if self.region == "US":
            if not os.path.isdir(camels_config["CAMELS_DIR"]):
                os.makedirs(camels_config["CAMELS_DIR"])
            [download_one_zip(attr_url, camels_config["CAMELS_DIR"]) for attr_url in
             camels_config["CAMELS_DOWNLOAD_URL_LST"] if
             not os.path.isfile(os.path.join(camels_config["CAMELS_DIR"], attr_url.split("/")[-1]))]
            print("The CAMELS_US data have been downloaded!")
        print("Please download it manually and put all files of a CAMELS dataset in the CAMELS_DIR directory.")
        print("We unzip all files now.")
        for f_name in os.listdir(camels_config["CAMELS_DIR"]):
            if fnmatch.fnmatch(f_name, '*.zip'):
                unzip_dir = os.path.join(camels_config["CAMELS_DIR"], f_name[0:-4])
                file_name = os.path.join(camels_config["CAMELS_DIR"], f_name)
                unzip_nested_zip(file_name, unzip_dir)

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
            data = pd.read_csv(camels_file, sep=';', dtype={"gauge_id": str, "huc_02": str})
        elif self.region == "AUS":
            data = pd.read_csv(camels_file, sep=',', dtype={"station_id": str})
        elif self.region == "BR":
            data = pd.read_csv(camels_file, sep='\s+', dtype={"gauge_id": str})
        elif self.region == "CL":
            data = pd.read_csv(camels_file, sep='\t', index_col=0)
        elif self.region == "GB":
            data = pd.read_csv(camels_file, sep=',', dtype={"gauge_id": str})
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)
        return data

    def get_constant_cols(self) -> np.array:
        """
        all readable attrs in CAMELS

        Returns
        -------
        np.array
            attribute types
        """
        data_folder = self.data_source_description["CAMELS_ATTR_DIR"]
        if self.region == "US":
            var_dict = dict()
            var_lst = list()
            key_lst = self.data_source_description["CAMELS_ATTR_KEY_LST"]
            for key in key_lst:
                data_file = os.path.join(data_folder, 'camels_' + key + '.txt')
                data_temp = pd.read_csv(data_file, sep=';')
                var_lst_temp = list(data_temp.columns[1:])
                var_dict[key] = var_lst_temp
                var_lst.extend(var_lst_temp)
            return np.array(var_lst)
        elif self.region == "AUS":
            attr_all_file = os.path.join(self.data_source_description["CAMELS_DIR"],
                                         "CAMELS_AUS_Attributes-Indices_MasterTable.csv")
            camels_aus_attr_indices_data = pd.read_csv(attr_all_file, sep=',')
            # exclude station id
            return camels_aus_attr_indices_data.columns.values[1:]
        elif self.region == "BR":
            var_dict = dict()
            var_lst = list()
            key_lst = self.data_source_description["CAMELS_ATTR_KEY_LST"]
            for key in key_lst:
                data_file = os.path.join(data_folder, 'camels_br_' + key + '.txt')
                data_temp = pd.read_csv(data_file, sep='\s+')
                var_lst_temp = list(data_temp.columns[1:])
                var_dict[key] = var_lst_temp
                var_lst.extend(var_lst_temp)
            return np.array(var_lst)
        elif self.region == "CL":
            camels_cl_attr_data = self.camels_sites
            # exclude station id
            return camels_cl_attr_data.index.values
        elif self.region == "GB":
            var_dict = dict()
            var_lst = list()
            key_lst = self.data_source_description["CAMELS_ATTR_KEY_LST"]
            for key in key_lst:
                data_file = os.path.join(data_folder, "CAMELS_GB_" + key + '_attributes.csv')
                data_temp = pd.read_csv(data_file, sep=',')
                var_lst_temp = list(data_temp.columns[1:])
                var_dict[key] = var_lst_temp
                var_lst.extend(var_lst_temp)
            return np.array(var_lst)
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)

    def get_relevant_cols(self) -> np.array:
        """
        all readable forcing types

        Returns
        -------
        np.array
            forcing types
        """
        if self.region == "US":
            return np.array(["dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"])
        elif self.region == "AUS":
            forcing_types = []
            for root, dirs, files in os.walk(self.data_source_description["CAMELS_FORCING_DIR"]):
                if root == self.data_source_description["CAMELS_FORCING_DIR"]:
                    continue
                for file in files:
                    forcing_types.append(file[:-4])
            return np.array(forcing_types)
        elif self.region == "BR":
            return np.array([forcing_dir.split(os.sep)[-1][13:] for forcing_dir in
                             self.data_source_description["CAMELS_FORCING_DIR"]])
        elif self.region == "CL":
            return np.array(["_".join(forcing_dir.split(os.sep)[-1].split("_")[2:]) for forcing_dir in
                             self.data_source_description["CAMELS_FORCING_DIR"]])
        elif self.region == "GB":
            return np.array(["precipitation", "pet", "temperature", "peti", "humidity", "shortwave_rad", "longwave_rad",
                             "windspeed"])
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)

    def get_target_cols(self) -> np.array:
        """
        For CAMELS, the target vars are streamflows

        Returns
        -------
        np.array
            streamflow types
        """
        if self.region == "US":
            return np.array(["usgsFlow"])
        elif self.region == "AUS":
            # QualityCodes are not streamflow data.
            # MLd means "1 Megaliters Per Day"; 1 MLd = 0.011574074074074 cubic-meters-per-second
            # mmd means "mm/day"
            return np.array(
                ["streamflow_MLd", "streamflow_MLd_inclInfilled", "streamflow_mmd", "streamflow_QualityCodes"])
        elif self.region == "BR":
            return np.array(
                [flow_dir.split(os.sep)[-1][13:] for flow_dir in self.data_source_description["CAMELS_FLOW_DIR"]])
        elif self.region == "CL":
            return np.array(
                [flow_dir.split(os.sep)[-1][11:] for flow_dir in self.data_source_description["CAMELS_FLOW_DIR"]])
        elif self.region == "GB":
            return np.array(["discharge_spec", "discharge_vol"])
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)

    def get_other_cols(self) -> dict:
        return {"FDC": {"time_range": ["1980-01-01", "2000-01-01"], "quantile_num": 100}}

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
        if self.region in ["BR", "GB", "US"]:
            return self.camels_sites["gauge_id"].values
        elif self.region == "AUS":
            return self.camels_sites["station_id"].values
        elif self.region == "CL":
            station_ids = self.camels_sites.columns.values
            # for 7-digit id, replace the space with 0 to get a 8-digit id
            cl_station_ids = [station_id.split(" ")[-1].zfill(8) for station_id in station_ids]
            return np.array(cl_station_ids)
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)

    def read_usgs_gage(self, usgs_id, t_range):
        """
        read streamflow data of a station from CAMELS-US

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
        print("reading %s streamflow data", usgs_id)
        gage_id_df = self.camels_sites
        huc = gage_id_df[gage_id_df["gauge_id"] == usgs_id]["huc_02"].values[0]
        usgs_file = os.path.join(self.data_source_description["CAMELS_FLOW_DIR"], huc, usgs_id + '_streamflow_qc.txt')
        data_temp = pd.read_csv(usgs_file, sep=r'\s+', header=None)
        obs = data_temp[4].values
        obs[obs < 0] = np.nan
        t_lst = hydro_utils.t_range_days(t_range)
        nt = t_lst.shape[0]
        if len(obs) != nt:
            out = np.full([nt], np.nan)
            df_date = data_temp[[1, 2, 3]]
            df_date.columns = ['year', 'month', 'day']
            date = pd.to_datetime(df_date).values.astype('datetime64[D]')
            [C, ind1, ind2] = np.intersect1d(date, t_lst, return_indices=True)
            out[ind2] = obs[ind1]
        else:
            out = obs
        return out

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
        dir_ = [flow_dir for flow_dir in self.data_source_description["CAMELS_FLOW_DIR"] if flow_type in flow_dir][0]
        if flow_type == "streamflow_mm_selected_catchments":
            flow_type = "streamflow_mm"
        elif flow_type == "streamflow_simulated":
            flow_type = "simulated_streamflow"
        gage_file = os.path.join(dir_, gage_id + "_" + flow_type + ".txt")
        data_temp = pd.read_csv(gage_file, sep=r'\s+')
        obs = data_temp.iloc[:, 3].values
        obs[obs < 0] = np.nan
        df_date = data_temp[['year', 'month', 'day']]
        date = pd.to_datetime(df_date).values.astype('datetime64[D]')
        out = time_intersect_dynamic_data(obs, date, t_range)
        return out

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
        gage_file = os.path.join(self.data_source_description["CAMELS_FLOW_DIR"],
                                 "CAMELS_GB_hydromet_timeseries_" + gage_id + "_19701001-20150930.csv")
        data_temp = pd.read_csv(gage_file, sep=",")
        obs = data_temp[var_type].values
        if var_type in ["discharge_spec", "discharge_vol"]:
            obs[obs < 0] = np.nan
        date = pd.to_datetime(data_temp["date"]).values.astype('datetime64[D]')
        out = time_intersect_dynamic_data(obs, date, t_range)
        return out

    def read_target_cols(self, gage_id_lst: Union[list, np.array] = None, t_range: list = None,
                         target_cols: Union[list, np.array] = None, **kwargs) -> np.array:
        """
        read target values; for CAMELS, they are streamflows
        default target_cols is an one-value list

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
        y = np.empty([len(gage_id_lst), nt, nf])
        if self.region == "US":
            for k in range(len(gage_id_lst)):
                data_obs = self.read_usgs_gage(gage_id_lst[k], t_range)
                # For CAMELS-US, only ["usgsFlow"]
                y[k, :, 0] = data_obs
        elif self.region == "AUS":
            for k in range(len(target_cols)):
                flow_data = pd.read_csv(
                    os.path.join(self.data_source_description["CAMELS_FLOW_DIR"], target_cols[k] + ".csv"))
                df_date = flow_data[['year', 'month', 'day']]
                date = pd.to_datetime(df_date).values.astype('datetime64[D]')
                [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
                chosen_data = flow_data[gage_id_lst].values[ind1, :]
                y[:, :, k] = chosen_data.T
        elif self.region == "BR":
            for j in range(len(target_cols)):
                for k in range(len(gage_id_lst)):
                    data_obs = self.read_br_gage_flow(gage_id_lst[k], t_range, target_cols[j])
                    y[k, :, j] = data_obs
        elif self.region == "CL":
            for k in range(len(target_cols)):
                if target_cols[k] == "streamflow_m3s":
                    flow_data = pd.read_csv(os.path.join(self.data_source_description["CAMELS_FLOW_DIR"][0],
                                                         "2_CAMELScl_streamflow_m3s.txt"), sep="\t", index_col=0)
                elif target_cols[k] == "streamflow_mm":
                    flow_data = pd.read_csv(os.path.join(self.data_source_description["CAMELS_FLOW_DIR"][1],
                                                         "3_CAMELScl_streamflow_mm.txt"), sep="\t", index_col=0)
                else:
                    raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)
                date = pd.to_datetime(flow_data.index.values).values.astype('datetime64[D]')
                [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
                station_ids = self.read_object_ids()
                assert (all(x < y for x, y in zip(station_ids, station_ids[1:])))
                [s, ind3, ind4] = np.intersect1d(station_ids, gage_id_lst, return_indices=True)
                chosen_data = flow_data.iloc[ind1, ind3].replace("\s+", np.nan, regex=True)
                y[:, :, k] = chosen_data.values.T
        elif self.region == "GB":
            for j in range(len(target_cols)):
                for k in range(len(gage_id_lst)):
                    data_obs = self.read_gb_gage_flow_forcing(gage_id_lst[k], t_range, target_cols[j])
                    y[k, :, j] = data_obs
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)
        return y

    def read_forcing_gage(self, usgs_id, var_lst, t_range_list, forcing_type='daymet'):
        # data_source = daymet or maurer or nldas
        print("reading %s forcing data", usgs_id)
        gage_id_df = self.camels_sites
        huc = gage_id_df[gage_id_df["gauge_id"] == usgs_id]["huc_02"].values[0]

        data_folder = self.data_source_description["CAMELS_FORCING_DIR"]
        if forcing_type == 'daymet':
            temp_s = 'cida'
        else:
            temp_s = forcing_type
        data_file = os.path.join(data_folder, forcing_type, huc, '%s_lump_%s_forcing_leap.txt' % (usgs_id, temp_s))
        data_temp = pd.read_csv(data_file, sep=r'\s+', header=None, skiprows=4)
        forcing_lst = ["Year", "Mnth", "Day", "Hr", "dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"]
        df_date = data_temp[[0, 1, 2]]
        df_date.columns = ['year', 'month', 'day']
        date = pd.to_datetime(df_date).values.astype('datetime64[D]')

        nf = len(var_lst)
        [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
        nt = c.shape[0]
        out = np.empty([nt, nf])

        for k in range(nf):
            ind = forcing_lst.index(var_lst[k])
            out[:, k] = data_temp[ind].values[ind1]
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
        dir_ = [_dir for _dir in self.data_source_description["CAMELS_FORCING_DIR"] if var_type in _dir][0]
        if var_type in ['temperature_min_cpc', 'temperature_mean_cpc', 'temperature_max_cpc']:
            var_type = var_type[:-4]
        gage_file = os.path.join(dir_, gage_id + "_" + var_type + ".txt")
        data_temp = pd.read_csv(gage_file, sep=r'\s+')
        obs = data_temp.iloc[:, 3].values
        df_date = data_temp[['year', 'month', 'day']]
        date = pd.to_datetime(df_date).values.astype('datetime64[D]')
        out = time_intersect_dynamic_data(obs, date, t_range)
        return out

    def read_relevant_cols(self, gage_id_lst: list = None, t_range: list = None, var_lst: list = None,
                           forcing_type="daymet") -> np.array:
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
        x = np.empty([len(gage_id_lst), nt, len(var_lst)])
        if self.region == "US":
            for k in range(len(gage_id_lst)):
                data = self.read_forcing_gage(gage_id_lst[k], var_lst, t_range_list, forcing_type=forcing_type)
                x[k, :, :] = data
        elif self.region == "AUS":
            for k in range(len(var_lst)):
                if "precipitation_" in var_lst[k]:
                    forcing_dir = os.path.join(self.data_source_description["CAMELS_FORCING_DIR"],
                                               "01_precipitation_timeseries")
                elif "et_" in var_lst[k] or "evap_" in var_lst[k]:
                    forcing_dir = os.path.join(self.data_source_description["CAMELS_FORCING_DIR"],
                                               "02_EvaporativeDemand_timeseries")
                else:
                    if "_AWAP" in var_lst[k]:
                        forcing_dir = os.path.join(self.data_source_description["CAMELS_FORCING_DIR"], "03_Other",
                                                   "AWAP")
                    elif "_SILO" in var_lst[k]:
                        forcing_dir = os.path.join(self.data_source_description["CAMELS_FORCING_DIR"], "03_Other",
                                                   "SILO")
                    else:
                        raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)
                forcing_data = pd.read_csv(os.path.join(forcing_dir, var_lst[k] + ".csv"))
                df_date = forcing_data[['year', 'month', 'day']]
                date = pd.to_datetime(df_date).values.astype('datetime64[D]')
                [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
                chosen_data = forcing_data[gage_id_lst].values[ind1, :]
                x[:, :, k] = chosen_data.T
        elif self.region == "BR":
            for j in range(len(var_lst)):
                for k in range(len(gage_id_lst)):
                    data_obs = self.read_br_basin_forcing(gage_id_lst[k], t_range, var_lst[j])
                    x[k, :, j] = data_obs
        elif self.region == "CL":
            for k in range(len(var_lst)):
                for tmp in os.listdir(self.data_source_description["CAMELS_DIR"]):
                    if fnmatch.fnmatch(tmp, '*' + var_lst[k]):
                        tmp_ = os.path.join(self.data_source_description["CAMELS_DIR"], tmp)
                        if os.path.isdir(tmp_):
                            forcing_file = os.path.join(tmp_, os.listdir(tmp_)[0])
                forcing_data = pd.read_csv(forcing_file, sep="\t", index_col=0)
                date = pd.to_datetime(forcing_data.index.values).values.astype('datetime64[D]')
                [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
                station_ids = self.read_object_ids()
                assert (all(x < y for x, y in zip(station_ids, station_ids[1:])))
                [s, ind3, ind4] = np.intersect1d(station_ids, gage_id_lst, return_indices=True)
                chosen_data = forcing_data.iloc[ind1, ind3].replace("\s+", np.nan, regex=True)
                x[:, :, k] = chosen_data.values.T
        elif self.region == "GB":
            for j in range(len(var_lst)):
                for k in range(len(gage_id_lst)):
                    data_forcing = self.read_gb_gage_flow_forcing(gage_id_lst[k], t_range, var_lst[j])
                    x[k, :, j] = data_forcing
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)
        return x

    def read_attr_all(self):
        data_folder = self.data_source_description["CAMELS_ATTR_DIR"]
        key_lst = self.data_source_description["CAMELS_ATTR_KEY_LST"]
        f_dict = dict()  # factorize dict
        var_dict = dict()
        var_lst = list()
        out_lst = list()
        gage_dict = self.camels_sites
        if self.region == "US":
            camels_str = "camels_"
            sep_ = ';'
        elif self.region == "BR":
            camels_str = "camels_br_"
            sep_ = '\s+'
        elif self.region == "GB":
            camels_str = "CAMELS_GB_"
            sep_ = ','
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)
        for key in key_lst:
            data_file = os.path.join(data_folder, camels_str + key + '.txt')
            if self.region == "GB":
                data_file = os.path.join(data_folder, camels_str + key + '_attributes.csv')
            data_temp = pd.read_csv(data_file, sep=sep_)
            var_lst_temp = list(data_temp.columns[1:])
            var_dict[key] = var_lst_temp
            var_lst.extend(var_lst_temp)
            k = 0
            n_gage = len(gage_dict['gauge_id'].values)
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
            attr_all_file = os.path.join(self.data_source_description["CAMELS_DIR"],
                                         "CAMELS_AUS_Attributes-Indices_MasterTable.csv")
            all_attr = pd.read_csv(attr_all_file, sep=',')
        elif self.region == "CL":
            attr_all_file = os.path.join(self.data_source_description["CAMELS_ATTR_DIR"], "1_CAMELScl_attributes.txt")
            all_attr_tmp = pd.read_csv(attr_all_file, sep='\t', index_col=0)
            all_attr = pd.DataFrame(all_attr_tmp.values.T, index=all_attr_tmp.columns, columns=all_attr_tmp.index)
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)
        # gage_all_attr = all_attr[all_attr['station_id'].isin(gage_id_lst)]
        var_lst = self.get_constant_cols().tolist()
        data_temp = all_attr[var_lst]
        # for factorized data, we need factorize all gages' data to keep the factorized number same all the time
        n_gage = len(self.read_object_ids())
        out_temp = np.full([n_gage, len(var_lst)], np.nan)
        f_dict = {}
        out_lst = []
        k = 0
        for field in var_lst:
            if is_string_dtype(data_temp[field]):
                value, ref = pd.factorize(data_temp[field], sort=True)
                out_temp[:, k] = value
                f_dict[field] = ref.tolist()
            elif is_numeric_dtype(data_temp[field]):
                out_temp[:, k] = data_temp[field].values
            k = k + 1
            out_lst.append(out_temp)
        out = np.concatenate(out_lst, 1)
        # keep same format with CAMELS_US
        return out, var_lst, None, f_dict

    def read_constant_cols(self, gage_id_lst=None, var_lst=None, is_return_dict=False):
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
        np.array
            if attr var type is str, return factorized data
        """
        if self.region in ["BR", "GB", "US"]:
            attr_all, var_lst_all, var_dict, f_dict = self.read_attr_all()
        elif self.region == "AUS" or self.region == "CL":
            attr_all, var_lst_all, var_dict, f_dict = self.read_attr_all_in_one_file()
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)
        ind_var = list()
        for var in var_lst:
            ind_var.append(var_lst_all.index(var))
        id_lst_all = self.read_object_ids()
        # TODO: Notice the sequence of station ids !!!!!!
        c, ind_grid, ind2 = np.intersect1d(id_lst_all, gage_id_lst, return_indices=True)
        temp = attr_all[ind_grid, :]
        out = temp[:, ind_var]
        if is_return_dict:
            return out, var_dict, f_dict
        else:
            return out

    def read_basin_area(self, object_ids) -> np.array:
        if self.region == "US":
            return self.read_constant_cols(object_ids, ['area_gages2'], is_return_dict=False)
        elif self.region == "AUS":
            return self.read_constant_cols(object_ids, ['catchment_area'], is_return_dict=False)
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)

    def read_mean_prep(self, object_ids) -> np.array:
        if self.region == "US" or self.region == "AUS":
            return self.read_constant_cols(object_ids, ['p_mean'], is_return_dict=False)
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)

    def read_other_cols(self, object_ids=None, other_cols: dict = None, **kwargs):
        # TODO: FDC for test period should keep same with that in training period
        out_dict = {}
        for key, value in other_cols.items():
            if key == "FDC":
                assert "time_range" in value.keys()
                if "quantile_num" in value.keys():
                    quantile_num = value["quantile_num"]
                    out = cal_fdc(self.read_target_cols(object_ids, value["time_range"], "usgsFlow"),
                                  quantile_num=quantile_num)
                else:
                    out = cal_fdc(self.read_target_cols(object_ids, value["time_range"], "usgsFlow"))
            else:
                raise NotImplementedError("No this item yet!!")
            out_dict[key] = out
        return out_dict
