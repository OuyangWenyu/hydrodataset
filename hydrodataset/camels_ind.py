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
import json
import warnings


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
            "streamflow_observed.csv",
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
            "name",  # gauge metadata
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
        return np.array(["streamflow_observed"])

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

    def read_ind_gage_forcing(self, gage_id, t_range, var_type):
        """
        Read gage's streamflow or forcing from CAMELS-IND

        Parameters
        ----------
        gage_id
            the station id
        t_range
            the time range, for example, ["1980-01-01", "2020-12-31"]
        var_type
            forcing type: "prcp(mm/day)", "tmax(C)", "tmin(C)", "tavg(C)", "srad_lw(w/m2)", "srad_sw(w/m2)", "wind_u(m/s)",
            "wind_v(m/s)", "wind(m/s)", "rel_hum(%)", "pet(mm/day)", "pet_gleam(mm/day)", "aet_gleam(mm/day)", "evap_canopy(kg/m2/s)",
            "evap_surface(kg/m2/s)", "sm_lvl1(kg/m2)", "sm_lvl2(kg/m2)", "sm_lvl3(kg/m2)", "sm_lvl4(kg/m2)"

        Returns
        -------
        np.array
            streamflow or forcing data of one station for a given time range
        """
        logging.debug("reading %s forcing data", gage_id)
        gage_file = os.path.join(
            self.data_source_description["CAMELS_FORCING_DIR"],
            gage_id + ".csv",
        )
        data_temp = pd.read_csv(gage_file, sep=",")
        obs = data_temp[var_type].values
        df_date = data_temp[["year", "month", "day"]]
        date = pd.to_datetime(df_date).values.astype("datetime64[D]")
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
        for k in tqdm(
            range(len(target_cols)), desc="Read streamflow data of CAMELS-IND"
        ):
            if target_cols[k] == "streamflow_observed":
                flow_data = pd.read_csv(
                    os.path.join(
                        self.data_source_description["CAMELS_FLOW_DIR"]
                    ),
                    sep=",",
                )
            else:
                raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)

            df_date = flow_data[["year", "month", "day"]]
            date = pd.to_datetime(df_date).values.astype("datetime64[D]")
            [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)   #用y与x对比，以x为主。返回的是[相交的元素, 相交的元素在x中的位置，相交的元素在y中的位置]。
            station_ids = [id_.zfill(5) for id_ in flow_data.columns.values]
            # assert all(x < y for x, y in zip(station_ids, station_ids[1:]))  # what's mean?
            ind3 = [station_ids.index(tmp) for tmp in gage_id_lst]
            # to guarantee the sequence is not changed we don't use np.intersect1d
            chosen_data = flow_data.iloc[ind1, ind3]
            chosen_data = chosen_data.astype(float)
            chosen_data[chosen_data < 0] = np.nan
            y[:, ind2, k] = chosen_data.values.T

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
                data_forcing = self.read_ind_gage_forcing(
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


    def cache_forcing_np_json(self):
        """
        Save all basin-forcing data in a numpy array file in the cache directory.

        Because it takes much time to read data from csv files,
        it is a good way to cache data as a numpy file to speed up the reading.
        In addition, we need a document to explain the meaning of all dimensions.

        """
        cache_npy_file = CACHE_DIR.joinpath("camels_ind_forcing.npy")
        json_file = CACHE_DIR.joinpath("camels_ind_forcing.json")
        variables = self.get_relevant_cols()
        basins = self.sites["gauge_id"].values
        t_range = ["1980-01-01", "2020-12-31"]
        times = [
            hydro_time.t2str(tmp)
            for tmp in hydro_time.t_range_days(t_range).tolist()
        ]

        variables_list = self.delete_variables_unit(variables)  # delete the unit behind the variables name, e.g. 'prcp(mm/day)' -> 'prcp'
        data_info = collections.OrderedDict(
            {
                "dim": ["basin", "time", "variable"],
                "basin": basins.tolist(),
                "time": times,
                "variable": variables_list,
            }
        )
        with open(json_file, "w") as FP:
            json.dump(data_info, FP, indent=4)
        data = self.read_relevant_cols(
            gage_id_lst=basins.tolist(),
            t_range=t_range,
            var_lst=variables.tolist(),
        )
        np.save(cache_npy_file, data)

    def cache_streamflow_np_json(self):
        """
        Save all basins' streamflow data in a numpy array file in the cache directory
        """
        cache_npy_file = CACHE_DIR.joinpath("camels_ind_streamflow.npy")
        json_file = CACHE_DIR.joinpath("camels_ind_streamflow.json")
        variables = self.get_target_cols()
        basins = self.sites["gauge_id"].values
        t_range = ["1980-01-01", "2020-12-31"]
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

        attr_all, var_lst_all, var_dict, f_dict = self.read_attr_all()
        attrs_df = pd.DataFrame(data=attr_all[0:, 0:], columns=var_lst_all)

        # unify id to basin
        attrs_df.index.name = "basin"
        # We use xarray dataset to cache all data
        ds_from_df = attrs_df.to_xarray()
        units_dict = {
            "num_dams": "dimensionless",
            "res_store_sum": "10^3m^3",
            "n_dams": "dimensionless",
            "first_dam_year": "dimensionless",
            "last_dam_year": "dimensionless",
            "total_storage": "m^3",
            "reservoir_index": "dimensionless",
            "irrigation_frac": "dimensionless",
            "hydroelec_frac": "dimensionless",
            "drinking_frac": "dimensionless",
            "flood_frac": "dimensionless",
            "overflow_frac": "dimensionless",
            "navigation_frac": "dimensionless",
            "tailing_frac": "dimensionless",
            "pop_density_2000": "people/km^2",
            "pop_density_2005": "people/km^2",
            "pop_density_2010": "people/km^2",
            "pop_density_2015": "people/km^2",
            "pop_density_2020": "people/km^2",
            "urban_frac_1985": "dimensionless",
            "urban_frac_1995": "dimensionless",
            "urban_frac_2005": "dimensionless",
            "crops_frac_1985": "dimensionless",
            "crops_frac_1995": "dimensionless",
            "crops_frac_2005": "dimensionless",
            "p_mean": "mm/day",
            "p_max": "mm/day",
            "p_mean_anum": "mm",
            "p_monthly_variability": "dimensionless",
            "p_annual_variability": "dimensionless",
            "p_unif": "dimensionless",
            "high_prec_freq": "days/year",
            "high_prec_dur": "days",
            "max_high_prec_dur": "days",
            "high_prec_timing": "season",
            "low_prec_freq": "days/year",
            "low_prec_dur": "days",
            "max_low_prec_dur": "days",
            "low_prec_timing": "season",
            "asynchronicity": "dimensionless",
            "tmin_mean": "Celsius degree",
            "tmax_mean": "Celsius degree",
            "pet_mean": "mm/day",
            "pet_min": "mm/day",
            "pet_max": "mm/day",
            "pet_mean_anum": "mm",
            "pet_gleam_mean": "mm/day",
            "aet_gleam_mean": "mm/day",
            "evap_canopy_mean": "mm/day",
            "evap_canopy_min": "mm/day",
            "evap_canopy_max": "mm/day",
            "evap_canopy_anum": "mm",
            "evap_surface_mean": "mm/day",
            "evap_surface_min": "mm/day",
            "evap_surface_max": "mm/day",
            "evap_surface_anum": "mm",
            "aridity_p_pet": "dimensionless",
            "aridity_pet_aet": "dimensionless",
            "ai_mean": "dimensionless",
            "rel_hum_mean": "percent",
            "srad_lw_mean": "w/m^2",
            "srad_sw_mean": "w/m^2",
            "wind_mean": "m/s",
            "sm_lvl1_mean": "kg/m^2",
            "sm_lvl2_mean": "kg/m^2",
            "sm_lvl3_mean": "kg/m^2",
            "sm_lvl4_mean": "kg/m^2",
            "geol_porosity": "dimensionless",
            "geol_permeability": "m^2",
            "geol_class_1st": "dimensionless",
            "geol_class_1st_frac": "dimensionless",
            "geol_class_2nd": "dimensionless",
            "geol_class_2nd_frac": "dimensionless",
            "carb_rocks_frac": "dimensionless",
            "q_mean": "mm/day",
            "runoff_ratio": "dimensionless",
            "streamflow_elas": "dimensionless",
            "slope_fdc": "dimensionless",
            "bfi": "dimensionless",
            "q_cv": "percent",
            "q_10": "mm/day",
            "q_25": "mm/day",
            "q_50": "mm/day",
            "q_75": "mm/day",
            "q_90": "mm/day",
            "q_zero": "days/year",
            "q_low_days": "days",
            "freq_q_low": "days/year",
            "q_high_days": "days",
            "freq_q_high": "days/year",
            "annual_q": "MCM/year",
            "mean_anum_flow": "MCM/year",
            "cen_time": "Day",
            "gini_flow": "dimensionless",
            "annual_max_1day": "m^3/s",
            "annual_max_3day": "m^3/s",
            "annual_max_7day": "m^3/s",
            "annual_max_30day": "m^3/s",
            "annual_max_90day": "m^3/s",
            "annual_min_7day": "m^3/s",
            "month_1day_max": "calendar month",
            "month_1day_min": "calendar month",
            "doy_min_flow": "Day",
            "doy_max_flow": "Day",
            "doy_min_flow_7": "Day",
            "doy_max_flow_7": "Day",
            "mean_jan_flow": "MCM/month",
            "mean_feb_flow": "MCM/month",
            "mean_mar_flow": "MCM/month",
            "mean_apr_flow": "MCM/month",
            "mean_may_flow": "MCM/month",
            "mean_jun_flow": "MCM/month",
            "mean_jul_flow": "MCM/month",
            "mean_aug_flow": "MCM/month",
            "mean_sep_flow": "MCM/month",
            "mean_oct_flow": "MCM/month",
            "mean_nov_flow": "MCM/month",
            "mean_dec_flow": "MCM/month",
            "cv_jan_flow": "percent",
            "cv_feb_flow": "percent",
            "cv_mar_flow": "percent",
            "cv_apr_flow": "percent",
            "cv_may_flow": "percent",
            "cv_jun_flow": "percent",
            "cv_jul_flow": "percent",
            "cv_aug_flow": "percent",
            "cv_sep_flow": "percent",
            "cv_oct_flow": "percent",
            "cv_nov_flow": "percent",
            "cv_dec_flow": "percent",
            "mean_swmn_flow": "MCM/season",
            "mean_atmn_flow": "MCM/season",
            "mean_wint_flow": "MCM/season",
            "mean_sumr_flow": "MCM/season",
            "q_mean_swmn": "mm/day",
            "q_5_swmn": "mm/day",
            "q_25_swmn": "mm/day",
            "q_50_swmn": "mm/day",
            "q_75_swmn": "mm/day",
            "q_95_swmn": "mm/day",
            "rise_rate_mean": "m^3/s",
            "rise_rate_median": "m^3/s",
            "rise_days": "days/year",
            "fall_rate_mean": "m^3/s",
            "fall_rate_median": "m^3/s",
            "fall_days": "days/year",
            "num_hyd_alt": "dimensionless",
            "water_frac": "dimensionless",
            "trees_frac": "dimensionless",
            "flooded_veg_frac": "dimensionless",
            "crops_frac": "dimensionless",
            "built_area_frac": "dimensionless",
            "bare_frac": "dimensionless",
            "range_frac": "dimensionless",
            "dom_land_cover": "dimensionless",
            "dom_land_cover_frac": "dimensionless",
            "lai_mean": "dimensionless",
            "lai_min": "dimensionless",
            "lai_max": "dimensionless",
            "lai_diff": "dimensionless",
            "soil_depth": "m",
            "soil_conductivity_top": "cm/day",
            "soil_conductivity_sub": "cm/day",
            "soil_awc_top": "m^3/m^3",
            "soil_awc_sub": "m^3/m^3",
            "soil_awsc_min": "mm/m",
            "soil_awsc_max": "mm/m",
            "soil_awsc_major": "mm/m",
            "sand_frac_top": "percent wt",
            "sand_frac_sub": "percent wt",
            "silt_frac_top": "percent wt",
            "silt_frac_sub": "percent wt",
            "clay_frac_top": "percent wt",
            "clay_frac_sub": "percent wt",
            "gravel_frac_top": "percent vol",
            "gravel_frac_sub": "percent vol",
            "bulkdens_top_major": "kg/dm^3",
            "bulkdense_top_mean": "kg/dm^3",
            "bulkdens_sub_major": "kg/dm^3",
            "bulkdens_sub_mean": "kg/dm^3",
            "org_carb_top_major": "percent wt",
            "org_carb_top_mean": "percent wt",
            "org_carb_sub_major": "percent wt",
            "org_carb_sub_mean": "percent wt",
            "organic_frac_top": "dimensionless",
            "organic_frac_sub": "dimensionless",
            "hsg_major": "dimensionless",
            "wtd": "m",
            "cwc_lat": "degree",
            "cwc_lon": "degree",
            "ghi_lat": "degree",
            "ghi_lon": "degree",
            "elev_mean": "m",
            "elev_median": "m",
            "elev_min": "m",
            "elev_max": "m",
            "slope_mean": "percent",
            "slope_median": "percent",
            "slope_min": "percent",
            "slope_max": "percent",
            "cwc_area": "km^2",
            "ghi_area": "km^2",
            "gauge_elevation": "m",
            "dpsbar": "m/km",
        }

        # Assign units to the variables in the Dataset
        for var_name in units_dict:
            if var_name in ds_from_df.data_vars:
                ds_from_df[var_name].attrs["units"] = units_dict[var_name]

        return ds_from_df

    def cache_forcing_xrdataset(self):
        """Save all basin-forcing data in a netcdf file in the cache directory.

        """
        cache_npy_file = CACHE_DIR.joinpath("camels_ind_forcing.npy")
        json_file = CACHE_DIR.joinpath("camels_ind_forcing.json")
        if (not os.path.isfile(cache_npy_file)) or (not os.path.isfile(json_file)):
            self.cache_forcing_np_json()
        forcing = np.load(cache_npy_file)
        with open(json_file, "r") as fp:
            forcing_dict = json.load(
                fp, object_pairs_hook=collections.OrderedDict
            )
        import pint_xarray

        basins = forcing_dict["basin"]
        times = pd.date_range(
            forcing_dict["time"][0], periods=len(forcing_dict["time"])
        )
        variables = forcing_dict["variable"]

        units = ["mm/day","°C", "°C", "°C", "w/m^2", "w/m^2", "m/s", "m/s", "m/s", "%", "mm/day", "mm/day", "mm/day", "kg/m^2/s", "kg/m^2/s", "kg/m^2", "kg/m^2", "kg/m^2", "kg/m^2"]
        return xr.Dataset(
            data_vars={
                **{
                    variables[i]: (
                        ["basin", "time"],
                        forcing[:, :, i],
                        {"units": units[i]},
                    )
                    for i in range(len(variables))
                }
            },
            coords={
                "basin": basins,
                "time": times,
            },
            attrs={"forcing_type": "observation"},
        )

    def cache_streamflow_xrdataset(self):
        """Save all basins' streamflow data in a netcdf file in the cache directory

        """
        cache_npy_file = CACHE_DIR.joinpath("camels_ind_streamflow.npy")
        json_file = CACHE_DIR.joinpath("camels_ind_streamflow.json")
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
                # todo: what about ET?
                # "ET": (  # todo: where the ET data comes from?
                #     ["basin", "time"],
                #     streamflow[:, :, 1],
                #     {"units": "mm/day"},
                # ),
            },
            coords={
                "basin": basins,
                "time": times,
            },
        )

    def cache_xrdataset(self):
        """
        Save all data in a netcdf file in the cache directory

        """

        warnings.warn("Check you units of all variables")
        ds_attr = self.cache_attributes_xrdataset()
        ds_attr.to_netcdf(CACHE_DIR.joinpath("camelsind_attributes.nc"))
        ds_streamflow = self.cache_streamflow_xrdataset()
        ds_forcing = self.cache_forcing_xrdataset()
        ds = xr.merge([ds_streamflow, ds_forcing])
        ds.to_netcdf(CACHE_DIR.joinpath("camelsind_timeseries.nc"))
