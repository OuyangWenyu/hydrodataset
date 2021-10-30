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

            download_url_lst = [
                "https://ral.ucar.edu/sites/default/files/public/product-tool/camels-catchment-attributes-and-meteorology-for-large-sample-studies-dataset-downloads/camels_attributes_v2.0.zip",
                "https://ral.ucar.edu/sites/default/files/public/product-tool/camels-catchment-attributes-and-meteorology-for-large-sample-studies-dataset-downloads/basin_set_full_res.zip",
                "https://ral.ucar.edu/sites/default/files/public/product-tool/camels-catchment-attributes-and-meteorology-for-large-sample-studies-dataset-downloads/basin_timeseries_v1p2_metForcing_obsFlow.zip"]

            return collections.OrderedDict(CAMELS_DIR=camels_db, CAMELS_FLOW_DIR=flow_dir,
                                           CAMELS_FORCING_DIR=forcing_dir, CAMELS_FORCING_TYPE=forcing_types,
                                           CAMELS_ATTR_DIR=attr_dir, CAMELS_GAUGE_FILE=gauge_id_file,
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
            print("The CAMELS data have been downloaded!")
        else:
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
            key_lst = ['topo', 'clim', 'hydro', 'vege', 'soil', 'geol']
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
            return np.array(['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp'])
        elif self.region == "AUS":
            forcing_types = []
            for root, dirs, files in os.walk(self.data_source_description["CAMELS_FORCING_DIR"]):
                if root == self.data_source_description["CAMELS_FORCING_DIR"]:
                    continue
                for file in files:
                    forcing_types.append(file[:-4])
            return np.array(forcing_types)
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
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)

    def get_other_cols(self) -> dict:
        return {"FDC": {"time_range": ["1980-01-01", "2000-01-01"], "quantile_num": 100}}

    def read_object_ids(self) -> np.array:
        """
        read station ids

        Returns
        -------
        np.array
            gage/station ids
        """
        if self.region == "US":
            return self.camels_sites["gauge_id"].values
        elif self.region == "AUS":
            return self.camels_sites["station_id"].values
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
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)
        return x

    def read_attr_all(self):
        data_folder = self.data_source_description["CAMELS_ATTR_DIR"]
        f_dict = dict()  # factorize dict
        var_dict = dict()
        var_lst = list()
        out_lst = list()
        key_lst = ['topo', 'clim', 'hydro', 'vege', 'soil', 'geol']
        gage_dict = self.camels_sites
        for key in key_lst:
            data_file = os.path.join(data_folder, 'camels_' + key + '.txt')
            data_temp = pd.read_csv(data_file, sep=';')
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

    def read_attr_all_aus(self):
        """
        Read all attr data in CAMELS_AUS

        Returns
        -------
        np.array
            all attr data in CAMELS_AUS
        """
        attr_all_file = os.path.join(self.data_source_description["CAMELS_DIR"],
                                     "CAMELS_AUS_Attributes-Indices_MasterTable.csv")
        all_attr = pd.read_csv(attr_all_file, sep=',')
        # gage_all_attr = all_attr[all_attr['station_id'].isin(gage_id_lst)]
        var_lst = self.get_constant_cols().tolist()
        data_temp = all_attr[var_lst]
        # for factorized data, we need factorize all gages' data to keep the factorized number same all the time
        n_gage = len(self.camels_sites)
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
        gage_id_str = "gauge_id"
        if self.region == "US":
            attr_all, var_lst_all, var_dict, f_dict = self.read_attr_all()
        elif self.region == "AUS":
            attr_all, var_lst_all, var_dict, f_dict = self.read_attr_all_aus()
            gage_id_str = "station_id"
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)
        ind_var = list()
        for var in var_lst:
            ind_var.append(var_lst_all.index(var))
        gage_dict = self.camels_sites
        id_lst_all = gage_dict[gage_id_str].values
        # Notice the sequence of station ids !!!!!!
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
