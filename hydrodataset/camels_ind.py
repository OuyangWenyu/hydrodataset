import os
import logging
import collections
import pandas as pd
import numpy as np
from typing import Union
from tqdm import tqdm
import xarray as xr
from hydroutils import hydro_time, hydro_file
from hydrodataset import HydroDataset, CACHE_DIR, CAMELS_REGIONS
from hydrodataset.camels import Camels, time_intersect_dynamic_data
from pandas.api.types import is_string_dtype, is_numeric_dtype

CAMELS_NO_DATASET_ERROR_LOG = (
    "We cannot read this dataset now. Please check if you choose correctly:\n"
    + str(CAMELS_REGIONS)
)

class CamelsInd(Camels):
    def __init__(
        self,
        data_path = os.path.join("camels","camels_ind"),
        download = False,
        region: str = "IND",
    ):
        """
        Initialization for CAMELS-IND dataset

        Parameters
        ----------
        data_path
            where we put the dataset.
            we already set the ROOT directory for hydrodataset,
            so here just set it as a relative path,
            by default "camels/camels_ind"
        download
            if true, download, by default False
        region
            the default is CAMELS-IND
        """
        super().__init__(data_path,download,region)

    def set_data_source_describe(self) -> collections.OrderedDict:
        """
        the files in the dataset and their location in file system

        Returns
        -------
        collections.OrderedDict
            the description for a CAMELS-IND dataset
        """
        camels_db = self.data_source_dir

        if self.region == "IND":
            return self._set_data_source_camelsind_describe(camels_db)
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)

    def _set_data_source_camelsind_describe(self, camels_db):
        # shp file of basins
        camels_shp_file = camels_db.joinpath(
            "CAMELS_IND_All_Catchments",
            "shapefiles_catchment",
            "merged",
            "all_catchments.shp",
        )
        # streamflow
        flow_dir = camels_db.joinpath(
            "CAMELS_IND_All_Catchments",
            "streamflow_timeseries",
        )
        # forcing
        forcing_dir = camels_db.joinpath(
            "CAMELS_IND_All_Catchments",
            "catchment_mean_forcings",
        )
        forcing_types = ["observation"]
        # attr
        attr_dir = camels_db.joinpath(
            "CAMELS_IND_All_Catchments",
            "attributes_csv",
        )
        attr_key_lst = [
            "anth",
            "clim",
            "geol",
            "hydro",
            "land",
            # "name",  # gauge metadata
            "soil",
            "topo",
        ]
        gauge_id_file = attr_dir.joinpath("camels_ind_clim.csv")

        return collections.OrderedDict(
            CAMELS_DIR = camels_db,
            CAMELS_FLOW_DIR = flow_dir,
            CAMELS_FORCING_DIR = forcing_dir,
            CAMELS_FORCING_TYPE=forcing_types,
            CAMELS_ATTR_DIR = attr_dir,
            CAMELS_ATTR_KEY_LST = attr_key_lst,
            CAMELS_GAUGE_FILE = gauge_id_file,
            CAMELS_BASINS_SHP = camels_shp_file,
        )

    def read_site_info(self) -> pd.DataFrame:
        """
        Read the basic information of gages in a CAMELS-IND dataset.

        Returns
        -------
        pd.DataFrame
            basic info of gages
        """
        camels_file = self.data_source_description["CAMELS_GAUGE_FILE"]
        return pd.read_csv(camels_file,sep=",",dtype={"gauge_id": str})

    def get_constant_cols(self) -> np.ndarray:
        """
        all readable attrs in CAMELS-IND

        Returns
        -------
        np.ndarray
            attribute types
        """
        data_folder = self.data_source_description["CAMELS_ATTR_DIR"]
        return self._get_constant_cols_some(
            data_folder, "camels_ind_",".csv",","
        )

    def get_relevant_cols(self) -> np.ndarray:
        """
        all readable forcing types in CAMELS-IND

        Returns
        -------
        np.ndarray
            forcing types
        """
        return np.array(
            [
                "prcp(mm/day)",
                "tmax(C)",
                "tmin(C)",
                "tavg(C)",
                "srad_lw(w/m2)",
                "srad_sw(w/m2)",
                "wind_u(m/s)",
                "wind_v(m/s)",
                "wind(m/s)",
                "rel_hum(%)",
                "pet(mm/day)",
                "pet_gleam(mm/day)",
                "aet_gleam(mm/day)",
                "evap_canopy(kg/m2/s)",
                "evap_surface(kg/m2/s)",
                "sm_lvl1(kg/m2)",
                "sm_lvl2(kg/m2)",
                "sm_lvl3(kg/m2)",
                "sm_lvl4(kg/m2)",
            ]
        )

    def get_target_cols(self) -> np.ndarray:
        """
        For CAMELS-IND, the target vars are streamflows

        Returns
        -------
        np.ndarray
            streamflow types
        """
        return np.array(["discharge_vol"])  # todo: streamflow separate with forcing

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

    def read_ind_gage_flow_forcing(self, gage_id, t_range, var_type):
        """
        Read gage's streamflow or forcing from CAMELS-IND

        Parameters
        ----------
        gage_id
            the station id
        t_range
            the time range, for example, ["1980-01-01", "2020-12-31"]   todo: a problem for date format, ["1980,1,1", "2020,12,31"]
        var_type
            flow type: "discharge_vol", "discharge_spec"
            forcing type: "prcp(mm/day)", "tmax(C)", "tmin(C)", "tavg(C)", "srad_lw(w/m2)", "srad_sw(w/m2)", "wind_u(m/s)",
            "wind_v(m/s)", "wind(m/s)", "rel_hum(%)", "pet(mm/day)", "pet_gleam(mm/day)", "aet_gleam(mm/day)", "evap_canopy(kg/m2/s)",
            "evap_surface(kg/m2/s)", "sm_lvl1(kg/m2)", "sm_lvl2(kg/m2)", "sm_lvl3(kg/m2)", "sm_lvl4(kg/m2)"

        Returns
        -------
        np.array
            streamflow or forcing data of one station for a given time range
        """
        logging.debug("reading %s streamflow data", gage_id)
        gage_file = os.path.join(
            self.data_source_description["CAMELS_FORCING_DIR"],
            gage_id + ".csv",
        )
        data_temp = pd.read_csv(gage_file, sep=",")
        obs = data_temp[var_type].values
        if var_type in ["discharge_vol"]: # todo: streamflow separate with forcing
            obs[obs < 0] = np.nan
        date = pd.to_datetime(data_temp["date"]).values.astype("datetime64[D]")  #
        return time_intersect_dynamic_data(obs, date, t_range)

    def read_target_cols(
        self,
        gage_id_lst: Union[list, np.array] = None,
        t_range: list = None,
        target_cols: Union[list, np.array] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        read target values. for CAMELS-IND, they are streamflows.

        default target_cols is an one-value list
        Notice, the unit of target outputs in different regions are not totally same

        Parameters
        ----------
        gage_id_lst
            station ids
        t_range
            the time range, for example, ["1980-01-01", "2020-12-31"]
        target_cols
            the default is None, but we need at least one default target.
            For CAMELS-IND, it's ["discharge_vol"]
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
        for j in tqdm(
            range(len(target_cols)), desc="Read streamflow data of CAMELS-IND"
        ):
            for k in tqdm(range(len(gage_id_lst))):
                data_obs = self.read_ind_gage_flow_forcing(
                    gage_id_lst[k], t_range, target_cols[j]
                )
                y[k, :, j] = data_obs
        # Keep unit of streamflow unified: we use ft3/s here
        # other units are m3/s -> ft3/s
        y = y * 35.314666721489
        return y

    def read_relevant_cols(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        forcing_type="observation",
        **kwargs,
    ) -> np.ndarray:
        """
        Read forcing data

        Parameters
        ----------
        gage_id_lst
            station ids
        t_range
            the time range, for example, ["1980-01-01", "2020-12-31"]
        var_lst
            forcing variable type: "prcp(mm/day)", "tmax(C)", "tmin(C)", "tavg(C)", "srad_lw(w/m2)", "srad_sw(w/m2)", "wind_u(m/s)",
            "wind_v(m/s)", "wind(m/s)", "rel_hum(%)", "pet(mm/day)", "pet_gleam(mm/day)", "aet_gleam(mm/day)", "evap_canopy(kg/m2/s)",
            "evap_surface(kg/m2/s)", "sm_lvl1(kg/m2)", "sm_lvl2(kg/m2)", "sm_lvl3(kg/m2)", "sm_lvl4(kg/m2)"
        forcing_type
            support for CAMELS-IND, there are two types: observation, simulated
        Returns
        -------
        np.array
            forcing data
        """
        t_range_list = hydro_time.t_range_days(t_range)
        nt = t_range_list.shape[0]
        x = np.full([len(gage_id_lst), nt, len(var_lst)], np.nan)
        for j in tqdm(range(len(var_lst)), desc="Read forcing data of CAMELS-IND"):
            for k in tqdm(range(len(gage_id_lst))):
                data_forcing = self.read_ind_gage_flow_forcing(
                    gage_id_lst[k], t_range, var_lst[j]
                )
                x[k, :, j] = data_forcing
        return x

    def read_attr_all(self):
        """
         Read Attributes data

        """
        data_folder = self.data_source_description["CAMELS_ATTR_DIR"]
        key_lst = self.data_source_description["CAMELS_ATTR_KEY_LST"]
        f_dict = {}
        var_dict = {}
        var_lst = []
        out_lst = []
        gage_dict = self.sites
        camels_str = "camels_ind_"
        sep_ = ","
        for key in key_lst:
            data_file = os.path.join(data_folder, camels_str + key + ".csv")
            data_temp = pd.read_csv(data_file, sep=sep_)
            var_lst_temp = list(data_temp.columns[1:])
            var_dict[key] = var_lst_temp
            var_lst.extend(var_lst_temp)
            k = 0
            gage_id_key = "gauge_id"
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

    def read_constant_cols(
        self, gage_id_lst=None, var_lst=None, is_return_dict=False, **kwargs
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
            if true, return var_dict and f_dict for CAMELS-IND
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
        return self.read_constant_cols(gage_id_lst, ["cwc_area"], is_return_dict=False)

    def read_mean_prcp(self, gage_id_lst, unit="mm/d") -> xr.Dataset:
        """Read mean precipitation data

        Parameters
        ----------
        gage_id_lst : list
            station ids
        unit : str, optional
            the unit of p_mean, by default "mm/d"

        Returns
        -------
        xr.Dataset
            mean precipitation data
        """
        data = self.read_constant_cols(
            gage_id_lst,
            ["p_mean"],
            is_return_dict=False,
        )
        if unit in ["mm/d", "mm/day"]:
            converted_data = data
        elif unit in ["mm/h", "mm/hour"]:
            converted_data = data / 24
        elif unit in ["mm/3h", "mm/3hour"]:
            converted_data = data / 8
        elif unit in ["mm/8d", "mm/8day"]:
            converted_data = data * 8
        else:
            raise ValueError(
                "unit must be one of ['mm/d', 'mm/day', 'mm/h', 'mm/hour', 'mm/3h', 'mm/3hour', 'mm/8d', 'mm/8day']"
            )
        return converted_data
