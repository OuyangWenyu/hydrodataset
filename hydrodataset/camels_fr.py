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

class CamelsFr(Camels):
    def __init__(
        self,
        data_path = os.path.join("camels","camels_fr"),
        download = False,
        region: str = "FR",
    ):
        """
        Initialization for CAMELS-FR dataset

        Parameters
        ----------
        data_path
            where we put the dataset.
            we already set the ROOT directory for hydrodataset,
            so here just set it as a relative path,
            by default "camels/camels_fr"
        download
            if true, download, by default False
        region
            the default is CAMELS-FR
        """
        super().__init__(data_path,download,region)

    def set_data_source_describe(self) -> collections.OrderedDict:
        """
        the files in the dataset and their location in file system

        Returns
        -------
        collections.OrderedDict
            the description for a CAMELS-FR dataset
        """
        camels_db = self.data_source_dir

        if self.region == "FR":
            return self._set_data_source_camelsfr_describe(camels_db)
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)

    def _set_data_source_camelsfr_describe(self, camels_db):
        # shp file of basins
        camels_shp_file = camels_db.joinpath(
            "CAMELS_FR_geography",
            "CAMELS_FR_catchment_boundaries.gpkg", # todo: fr gives a gis database file, maybe need a manually transform.
        )
        # flow and forcing data are in a same file
        flow_dir = camels_db.joinpath(
            "CAMELS_FR_time_series",
            "daily",
        )
        forcing_dir = flow_dir
        # attr
        attr_dir1 = camels_db.joinpath(
            "CAMELS_FR_attributes",
            "static_attributes",
        )
        attr_dir2 = camels_db.joinpath(
            "CAMELS_FR_attributes",
            "time_series_statistics"
        )
        attr_dir = [attr_dir1, attr_dir2]
        attr_key_lst = [    # the commented attribution files have different number of rows with station number
            "geology",
            "human_influences_dams",
            "hydrogeology",
            "land_cover",
            # "site_general",   # metadata   "sit_area_hydro", hydrological catchment area
            # "soil_general",
            # "soil_quantiles",
            "station_general",  # metadata   "sta_area_snap"  topographic catchment area (INRAE's own computation)
            "topography_general",
            # "topography_quantiles",
            "climatic_statistics",      # time_series_statistics
            # "hydroclimatic_quantiles",
            # "hydroclimatic_regimes_daily",
            "hydroclimatic_statistics_joint_availability_yearly",
            # "hydroclimatic_statistics_timeseries_yearly",
            "hydrological_signatures",
            "hydrometry_statistics",
        ]
        gauge_id_file = attr_dir1.joinpath("CAMELS_FR_geology_attributes.csv")

        return collections.OrderedDict(
            CAMELS_DIR = camels_db,
            CAMELS_FLOW_DIR = flow_dir,
            CAMELS_FORCING_DIR = forcing_dir,
            CAMELS_ATTR_DIR = attr_dir,
            CAMELS_ATTR_KEY_LST = attr_key_lst,
            CAMELS_GAUGE_FILE = gauge_id_file,
            CAMELS_BASINS_SHP = camels_shp_file,
        )

    def read_site_info(self) -> pd.DataFrame:
        """
        Read the basic information of gages in a CAMELS-FR dataset.

        Returns
        -------
        pd.DataFrame
            basic info of gages
        """
        camels_file = self.data_source_description["CAMELS_GAUGE_FILE"]
        return pd.read_csv(camels_file,sep=";",dtype={"sta_code_h3": str})

    def get_constant_cols(self) -> np.ndarray:
        """
        all readable attrs in CAMELS-FR

        Returns
        -------
        np.ndarray
            attribute types
        """
        data_folder = self.data_source_description["CAMELS_ATTR_DIR"]
        return self._get_constant_cols_some(
            data_folder, "CAMELS_FR_","_attributes.csv",";"
        )

    def get_relevant_cols(self) -> np.ndarray:
        """
        all readable forcing types in CAMELS-FR

        Returns
        -------
        np.ndarray
            forcing types
        """
        return np.array(
            [
                "tsd_prec",
                "tsd_prec_solid_frac",
                "tsd_temp",
                "tsd_pet_ou",
                "tsd_pet_pe",
                "tsd_pet_pm",
                "tsd_wind",
                "tsd_humid",
                "tsd_rad_dli",
                "tsd_rad_ssi",
                "tsd_swi_gr",
                "tsd_swi_isba",
                "tsd_swe_isba",
                "tsd_temp_min",
                "tsd_temp_max",
            ]
        )

    def get_target_cols(self) -> np.ndarray:
        """
        For CAMELS-FR, the target vars are streamflows

        Returns
        -------
        np.ndarray
            streamflow types
        """
        return np.array(["tsd_q_l", "tsd_q_mm"])

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
        return self.sites["sta_code_h3"].values

    def read_fr_gage_flow_forcing(self, gage_id, t_range, var_type):
        """
        Read gage's streamflow or forcing from CAMELS-FR

        Parameters
        ----------
        gage_id
            the station id
        t_range
            the time range, for example, ["1970-01-01", "2022-01-01"]
        var_type
            flow type: "tsd_q_l", "tsd_q_mm"
            forcing type: "tsd_prec","tsd_prec_solid_frac","tsd_temp","tsd_pet_ou","tsd_pet_pe","tsd_pet_pm","tsd_wind",
            "tsd_humid","tsd_rad_dli","tsd_rad_ssi","tsd_swi_gr","tsd_swi_isba","tsd_swe_isba","tsd_temp_min","tsd_temp_max"

        Returns
        -------
        np.array
            streamflow or forcing data of one station for a given time range
        """
        logging.debug("reading %s streamflow data", gage_id)
        gage_file = os.path.join(
            self.data_source_description["CAMELS_FLOW_DIR"],
            "CAMELS_FR_tsd_" + gage_id + ".csv",
        )
        data_temp = pd.read_csv(gage_file, sep=";",header=7)  # no need the "skiprows"
        obs = data_temp[var_type].values
        # if var_type in ["tsd_sq_l", "tsd_q_mm"]:
        #     obs[obs < 0] = np.nan
        date = pd.to_datetime(pd.Series(data_temp["tsd_date"]),format="%Y%m%d").dt.strftime("%Y-%m-%d").values.astype("datetime64[D]")
        return time_intersect_dynamic_data(obs, date, t_range)

    def read_target_cols(
        self,
        gage_id_lst: Union[list, np.array] = None,
        t_range: list = None,
        target_cols: Union[list, np.array] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        read target values. for CAMELS-FR, they are streamflows.

        default target_cols is an one-value list
        Notice, the unit of target outputs in different regions are not totally same

        Parameters
        ----------
        gage_id_lst
            station ids
        t_range
            the time range, for example, ["1970-01-01", "2022-01-01"]
        target_cols
            the default is None, but we need at least one default target.
            For CAMELS-FR, it's ["tsd_q_l"]
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
            range(len(target_cols)), desc="Read streamflow data of CAMELS-FR"
        ):
            for k in tqdm(range(len(gage_id_lst))):
                data_obs = self.read_fr_gage_flow_forcing(
                    gage_id_lst[k], t_range, target_cols[j]
                )
                y[k, :, j] = data_obs
        # Keep unit of streamflow unified: we use ft3/s here
        # unit conversion  L/s -> ft3/s
        y = self.unit_convert_streamflow_Ltofoot3(y)
        return y

    def read_relevant_cols(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        forcing_type="nan",
        **kwargs,
    ) -> np.ndarray:
        """
        Read forcing data

        Parameters
        ----------
        gage_id_lst
            station ids
        t_range
            the time range, for example, ["1970-01-01", "2022-01-01"]
        var_lst
            forcing variable type: "tsd_prec","tsd_prec_solid_frac","tsd_temp","tsd_pet_ou","tsd_pet_pe","tsd_pet_pm","tsd_wind",
            "tsd_humid","tsd_rad_dli","tsd_rad_ssi","tsd_swi_gr","tsd_swi_isba","tsd_swe_isba","tsd_temp_min","tsd_temp_max"
        forcing_type
            support for CAMELS-FR, there are ** types:
        Returns
        -------
        np.array
            forcing data
        """
        t_range_list = hydro_time.t_range_days(t_range)
        nt = t_range_list.shape[0]
        x = np.full([len(gage_id_lst), nt, len(var_lst)], np.nan)
        for j in tqdm(range(len(var_lst)), desc="Read forcing data of CAMELS-FR"):
            for k in tqdm(range(len(gage_id_lst))):
                data_forcing = self.read_fr_gage_flow_forcing(
                    gage_id_lst[k], t_range, var_lst[j]
                )
                x[k, :, j] = data_forcing
        return x

    def read_attr_all(self):
        """
         Read Attributes data

        """
        data_folder1 = self.data_source_description["CAMELS_ATTR_DIR"][0]
        data_folder2 = self.data_source_description["CAMELS_ATTR_DIR"][1]
        key_lst = self.data_source_description["CAMELS_ATTR_KEY_LST"]
        f_dict = {}
        var_dict = {}
        var_lst = []
        out_lst = []
        gage_dict = self.sites
        camels_str = "CAMELS_FR_"
        sep_ = ";"
        for key in key_lst:
            # locate the attribute file
            data_file1 = os.path.join(data_folder1, camels_str + key + "_attributes.csv")
            data_file2 = os.path.join(data_folder2, camels_str + key + ".csv")
            if os.path.exists(data_file1):
                data_file = data_file1
            elif os.path.exists(data_file2):
                data_file = data_file2
            data_temp = pd.read_csv(data_file, sep=sep_)
            var_lst_temp = list(data_temp.columns[1:])
            var_dict[key] = var_lst_temp
            var_lst.extend(var_lst_temp)
            k = 0
            gage_id_key = "sta_code_h3"
            n_gage = len(gage_dict[gage_id_key].values)
            print(len(var_lst_temp))
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
            if true, return var_dict and f_dict for CAMELS-FR
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
        return self.read_attr_xrdataset(gage_id_lst, ["sta_area_snap"], is_return_dict=False)

    def read_mean_prcp(self, gage_id_lst, unit="mm/d") -> xr.Dataset:
        """Read mean precipitation data

        Parameters
        ----------
        gage_id_lst : list
            station ids
        unit : str, optional
            the unit of cli_prec_mean, by default "mm/d"

        Returns
        -------
        xr.Dataset
            mean precipitation data
        """
        data = self.read_attr_xrdataset(
            gage_id_lst,
            ["cli_prec_mean"],
            is_return_dict=False,
        )
        converted_data = self.unit_convert_mean_prcp(data, unit="mm/d")
        return converted_data

    def cache_forcing_np_json(self):
        """
        Save all basin-forcing data in a numpy array file in the cache directory.

        Because it takes much time to read data from csv files,
        it is a good way to cache data as a numpy file to speed up the reading.
        In addition, we need a document to explain the meaning of all dimensions.

        """
        cache_npy_file = CACHE_DIR.joinpath("camels_fr_forcing.npy")
        json_file = CACHE_DIR.joinpath("camels_fr_forcing.json")
        variables = self.get_relevant_cols()
        basins = self.sites["sta_code_h3"].values
        t_range = ["1970-01-01", "2022-01-01"]
        times = [
            hydro_time.t2str(tmp)
            for tmp in hydro_time.t_range_days(t_range).tolist()
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
            t_range=t_range,
            var_lst=variables.tolist(),
        )
        np.save(cache_npy_file, data)

    def cache_streamflow_np_json(self):
        """
        Save all basins' streamflow data in a numpy array file in the cache directory
        """
        cache_npy_file = CACHE_DIR.joinpath("camels_fr_streamflow.npy")
        json_file = CACHE_DIR.joinpath("camels_fr_streamflow.json")
        variables = self.get_target_cols()
        basins = self.sites["sta_code_h3"].values
        t_range = ["1970-01-01", "2022-01-01"]
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
        gage_dict = self.sites
        gage_id_key = "sta_code_h3"
        gage = gage_dict[gage_id_key].values
        attrs_df = pd.DataFrame(data=attr_all[0:, 0:], index=gage, columns=var_lst_all)

        # unify id to basin
        attrs_df.index.name = "basin"
        # We use xarray dataset to cache all data
        ds_from_df = attrs_df.to_xarray()
        units_dict = {
            "geo_dom_class": "dimensionless",
            "geo_su": "percent",
            "geo_ss": "percent",
            "geo_py": "percent",
            "geo_sm": "percent",
            "geo_sc": "percent",
            "geo_ev": "percent",
            "geo_va": "percent",
            "geo_vi": "percent",
            "geo_vb": "percent",
            "geo_pa": "percent",
            "geo_pi": "percent",
            "geo_pb": "percent",
            "geo_mt": "percent",
            "geo_wb": "percent",
            "geo_ig": "percent",
            "geo_nd": "percent",
            "dam_n": "dimensionless",
            "dam_volume": "Mm^3",
            "dam_influence": "mm",
            "hgl_krs_not_karstic": "percent",
            "hgl_krs_karstic": "percent",
            "hgl_krs_unknown": "percent",
            "hgl_thm_alluvial": "percent",
            "hgl_thm_sedimentary": "percent",
            "hgl_thm_bedrock": "percent",
            "hgl_thm_intense_folded": "percent",
            "hgl_thm_volcanism": "percent",
            "hgl_thm_unknown": "percent",
            "hgl_permeability": "log10(m^2)",
            "hgl_porosity": "dimensionless",
            "clc_2018_lvl1_dom_class": "dimensionless",
            "clc_2018_lvl1_1": "percent",
            "clc_2018_lvl1_2": "percent",
            "clc_2018_lvl1_3": "percent",
            "clc_2018_lvl1_4": "percent",
            "clc_2018_lvl1_5": "percent",
            "clc_2018_lvl1_na": "percent",
            "clc_2018_lvl2_dom_class": "dimensionless",
            "clc_2018_lvl2_11": "percent",
            "clc_2018_lvl2_12": "percent",
            "clc_2018_lvl2_13": "percent",
            "clc_2018_lvl2_14": "percent",
            "clc_2018_lvl2_21": "percent",
            "clc_2018_lvl2_22": "percent",
            "clc_2018_lvl2_23": "percent",
            "clc_2018_lvl2_24": "percent",
            "clc_2018_lvl2_31": "percent",
            "clc_2018_lvl2_32": "percent",
            "clc_2018_lvl2_33": "percent",
            "clc_2018_lvl2_41": "percent",
            "clc_2018_lvl2_42": "percent",
            "clc_2018_lvl2_51": "percent",
            "clc_2018_lvl2_52": "percent",
            "clc_2018_lvl2_na": "percent",
            "clc_2018_lvl3_dom_class": "dimensionless",
            "clc_2018_lvl3_111": "percent",
            "clc_2018_lvl3_112": "percent",
            "clc_2018_lvl3_121": "percent",
            "clc_2018_lvl3_122": "percent",
            "clc_2018_lvl3_123": "percent",
            "clc_2018_lvl3_124": "percent",
            "clc_2018_lvl3_131": "percent",
            "clc_2018_lvl3_132": "percent",
            "clc_2018_lvl3_133": "percent",
            "clc_2018_lvl3_141": "percent",
            "clc_2018_lvl3_142": "percent",
            "clc_2018_lvl3_211": "percent",
            "clc_2018_lvl3_212": "percent",
            "clc_2018_lvl3_213": "percent",
            "clc_2018_lvl3_221": "percent",
            "clc_2018_lvl3_222": "percent",
            "clc_2018_lvl3_223": "percent",
            "clc_2018_lvl3_231": "percent",
            "clc_2018_lvl3_241": "percent",
            "clc_2018_lvl3_242": "percent",
            "clc_2018_lvl3_243": "percent",
            "clc_2018_lvl3_244": "percent",
            "clc_2018_lvl3_311": "percent",
            "clc_2018_lvl3_312": "percent",
            "clc_2018_lvl3_313": "percent",
            "clc_2018_lvl3_321": "percent",
            "clc_2018_lvl3_322": "percent",
            "clc_2018_lvl3_323": "percent",
            "clc_2018_lvl3_324": "percent",
            "clc_2018_lvl3_331": "percent",
            "clc_2018_lvl3_332": "percent",
            "clc_2018_lvl3_333": "percent",
            "clc_2018_lvl3_334": "percent",
            "clc_2018_lvl3_335": "percent",
            "clc_2018_lvl3_411": "percent",
            "clc_2018_lvl3_412": "percent",
            "clc_2018_lvl3_421": "percent",
            "clc_2018_lvl3_422": "percent",
            "clc_2018_lvl3_423": "percent",
            "clc_2018_lvl3_511": "percent",
            "clc_2018_lvl3_512": "percent",
            "clc_2018_lvl3_521": "percent",
            "clc_2018_lvl3_522": "percent",
            "clc_2018_lvl3_523": "percent",
            "clc_2018_lvl3_na": "percent",
            "clc_1990_lvl1_dom_class": "dimensionless",
            "clc_1990_lvl1_1": "percent",
            "clc_1990_lvl1_2": "percent",
            "clc_1990_lvl1_3": "percent",
            "clc_1990_lvl1_4": "percent",
            "clc_1990_lvl1_5": "percent",
            "clc_1990_lvl1_na": "percent",
            "clc_1990_lvl2_dom_class": "dimensionless",
            "clc_1990_lvl2_11": "percent",
            "clc_1990_lvl2_12": "percent",
            "clc_1990_lvl2_13": "percent",
            "clc_1990_lvl2_14": "percent",
            "clc_1990_lvl2_21": "percent",
            "clc_1990_lvl2_22": "percent",
            "clc_1990_lvl2_23": "percent",
            "clc_1990_lvl2_24": "percent",
            "clc_1990_lvl2_31": "percent",
            "clc_1990_lvl2_32": "percent",
            "clc_1990_lvl2_33": "percent",
            "clc_1990_lvl2_41": "percent",
            "clc_1990_lvl2_42": "percent",
            "clc_1990_lvl2_51": "percent",
            "clc_1990_lvl2_52": "percent",
            "clc_1990_lvl2_na": "percent",
            "clc_1990_lvl3_dom_class": "dimensionless",
            "clc_1990_lvl3_111": "percent",
            "clc_1990_lvl3_112": "percent",
            "clc_1990_lvl3_121": "percent",
            "clc_1990_lvl3_122": "percent",
            "clc_1990_lvl3_123": "percent",
            "clc_1990_lvl3_124": "percent",
            "clc_1990_lvl3_131": "percent",
            "clc_1990_lvl3_132": "percent",
            "clc_1990_lvl3_133": "percent",
            "clc_1990_lvl3_141": "percent",
            "clc_1990_lvl3_142": "percent",
            "clc_1990_lvl3_211": "percent",
            "clc_1990_lvl3_212": "percent",
            "clc_1990_lvl3_213": "percent",
            "clc_1990_lvl3_221": "percent",
            "clc_1990_lvl3_222": "percent",
            "clc_1990_lvl3_223": "percent",
            "clc_1990_lvl3_231": "percent",
            "clc_1990_lvl3_241": "percent",
            "clc_1990_lvl3_242": "percent",
            "clc_1990_lvl3_243": "percent",
            "clc_1990_lvl3_244": "percent",
            "clc_1990_lvl3_311": "percent",
            "clc_1990_lvl3_312": "percent",
            "clc_1990_lvl3_313": "percent",
            "clc_1990_lvl3_321": "percent",
            "clc_1990_lvl3_322": "percent",
            "clc_1990_lvl3_323": "percent",
            "clc_1990_lvl3_324": "percent",
            "clc_1990_lvl3_331": "percent",
            "clc_1990_lvl3_332": "percent",
            "clc_1990_lvl3_333": "percent",
            "clc_1990_lvl3_334": "percent",
            "clc_1990_lvl3_335": "percent",
            "clc_1990_lvl3_411": "percent",
            "clc_1990_lvl3_412": "percent",
            "clc_1990_lvl3_421": "percent",
            "clc_1990_lvl3_422": "percent",
            "clc_1990_lvl3_423": "percent",
            "clc_1990_lvl3_511": "percent",
            "clc_1990_lvl3_512": "percent",
            "clc_1990_lvl3_521": "percent",
            "clc_1990_lvl3_522": "percent",
            "clc_1990_lvl3_523": "percent",
            "clc_1990_lvl3_na": "percent",
            "top_altitude_mean": "m.a.s.l.",
            "top_slo_mean": "degree",
            "top_dist_outlet_mean": "km",
            "top_itopo_mean": "dimensionless",
            "top_slo_ori_n": "percent",
            "top_slo_ori_ne": "percent",
            "top_slo_ori_e": "percent",
            "top_slo_ori_se": "percent",
            "top_slo_ori_s": "percent",
            "top_slo_ori_sw": "percent",
            "top_slo_ori_w": "percent",
            "top_slo_ori_nw": "percent",
            "top_drainage_density": "km/km^2",
            "top_mor_form_factor_horton": "dimensionless",
            "top_mor_form_factor_square": "dimensionless",
            "top_mor_shape_factor": "dimensionless",
            "top_mor_compact_coef": "dimensionless",
            "top_mor_circ_ratio": "dimensionless",
            "top_mor_elong_ratio_circ": "dimensionless",
            "top_mor_elong_ratio_catchment": "dimensionless",
            "top_mor_relief_ratio": "dimensionless",
            "top_slo_flat": "percent",
            "top_slo_gentle": "percent",
            "top_slo_moderate": "percent",
            "top_slo_strong": "percent",
            "top_slo_steep": "percent",
            "top_slo_very_steep": "percent",
            "cli_prec_mean": "mm/day",
            "cli_pet_ou_mean": "mm/day",
            "cli_pet_pe_mean": "mm/day",
            "cli_pet_pm_mean": "mm/day",
            "cli_prec_mean_yr": "mm/yr",
            "cli_pet_ou_yr": "mm/yr",
            "cli_pet_pe_yr": "mm/yr",
            "cli_pet_pm_yr": "mm/yr",
            "cli_temp_mean": "°C/day",
            "cli_psol_frac_safran": "dimensionless",
            "cli_psol_frac_berghuijs": "dimensionless",
            "cli_aridity_ou": "dimensionless",
            "cli_aridity_pe": "dimensionless",
            "cli_aridity_pm": "dimensionless",
            "cli_prec_season_temp": "dimensionless",
            "cli_prec_season_pet_ou": "dimensionless",
            "cli_prec_season_pet_pe": "dimensionless",
            "cli_prec_season_pet_pm": "dimensionless",
            "cli_assync_ou": "dimensionless",
            "cli_assync_pe": "dimensionless",
            "cli_assync_pm": "dimensionless",
            "cli_prec_intensity": "dimensionless",
            "cli_prec_max": "mm/day",
            "cli_prec_date_max": "dimensionless",
            "cli_prec_freq_high": "days/yr",
            "cli_prec_dur_high": "days",
            "cli_prec_timing_high": "season",
            "cli_prec_freq_low": "days/yr",
            "cli_prec_dur_low": "days",
            "cli_prec_timing_low": "season",
            "hcy_qnt_quant": "dimensionless",
            "hcy_qnt_q": "mm/day",
            "hcy_qnt_prec": "mm/day",
            "hcy_qnt_temp": "°C/day",
            "hcy_qnt_pet_ou": "mm/day",
            "hcy_qnt_pet_pe": "mm/day",
            "hcy_qnt_pet_pm": "mm/day",
            "hcy_reg_quant": "dimensionless",
            "hcy_reg_day": "dimensionless",
            "hcy_reg_q": "mm/day",
            "hcy_reg_prec": "mm/day",
            "hcy_reg_temp": "°C/day",
            "hcy_reg_pet_ou": "mm/day",
            "hcy_reg_pet_pe": "mm/day",
            "hcy_reg_pet_pm": "mm/day",
            "hyc_jay_prec_mean": "mm/yr",
            "hyc_jay_pet_ou": "mm/yr",
            "hyc_jay_pet_pe": "mm/yr",
            "hyc_jay_pet_pm": "mm/yr",
            "hyc_jay_ratio_prec_pet_ou": "dimensionless",
            "hyc_jay_ratio_prec_pet_pe": "dimensionless",
            "hyc_jay_ratio_prec_pet_pm": "dimensionless",
            "hyc_jay_ratio_q_prec": "dimensionless",
            "hcy_tsy_year": "yr",
            "hcy_tsy_q_qmna": "mm/month",
            "hcy_tsy_q_max_day": "mm/day",
            "hcy_tsy_prec_daily_max": "mm/day",
            "hcy_tsy_prec_season_pet_ou": "dimensionless",
            "hcy_tsy_prec_season_pet_pe": "dimensionless",
            "hcy_tsy_prec_season_pet_pm": "dimensionless",
            "hyd_q_mean": "mm/day",
            "hyd_q_mean_yr": "mm/yr",
            "hyd_stream_elas": "dimensionless",
            "hyd_slope_fdc": "dimensionless",
            "hyd_bfi_ladson": "dimensionless",
            "hyd_bfi_lfstat": "dimensionless",
            "hyd_bfi_pelletier_pet_ou": "dimensionless",
            "hyd_hfd_mean": "days",
            "hyd_q_freq_high": "days/yr",
            "hyd_q_dur_high": "days",
            "hyd_q_freq_low": "days/yr",
            "hyd_q_dur_low": "days",
            "hyd_q_freq_zero": "days/yr",
            "hyd_q_max": "mm/day",
            "hyd_q_date_max": "dimensionless",
            "hyd_q_qmna_min": "mm/month",
            "hyd_q_date_qmna": "dimensionless",
            "hym_q_date_start": "dimensionless",
            "hym_q_date_end": "dimensionless",
            "hym_q_na_period": "percent",
            "hym_q_na_total": "percent",
            "hym_q_n_year": "dimensionless",
            "hym_q_questionable": "percent",
            "hym_q_unqualified": "percent",
            "hym_q_anomaly_inrae": "percent",
            "hym_q_low_uncertainty_inrae": "dimensionless",
            # "sit_label": "dimensionless",       # "site_general"
            # "sit_mnemonic": "dimensionless",
            # "sit_label_usual": "dimensionless",
            # "sit_label_add": "dimensionless",
            # "sit_type": "dimensionless",
            # "sit_type_add": "dimensionless",
            # "sta_code_h2": "dimensionless",
            # "sit_test_site": "dimensionless",
            # "sit_comment": "dimensionless",
            # "sit_city": "dimensionless",
            # "sit_latitude": "°N or m",  # degree N
            # "sit_longitude": "°E or m",
            # "sit_crs": "dimensionless",
            # "sit_zone_hydro": "dimensionless",
            # "sit_section": "dimensionless",
            # "sit_entity": "dimensionless",
            # "sit_waterbody": "dimensionless",
            # "sit_watercourse_acc": "dimensionless",
            # "sit_altitude": "m.a.s.l.",
            # "sit_altitude_datum": "dimensionless",
            # "sit_area_hydro": "km^2",
            # "sit_area_topo": "km^2",
            # "sit_tz": "dimensionless",
            # "sit_kp_up": "m",
            # "sit_kp_down": "m",
            # "sit_flood_duration": "dimensionless",
            # "sit_status": "dimensionless",
            # "sit_publication_rights": "dimensionless",
            # "sit_month1_low_water": "dimensionless",
            # "sit_month1_year": "dimensionless",
            # "sit_impact": "dimensionless",
            # "sit_section_vigilance": "dimensionless",
            # "sit_date_start": "dimensionless",
            # "sit_comment_impact_gene": "dimensionless",
            # "sit_date_update": "dimensionless",
            "sta_label": "dimensionless",
            "sta_label_add": "dimensionless",
            "sta_type": "dimensionless",
            "sta_test_station": "dimensionless",
            "sta_monitor": "dimensionless",
            "sta_code_h2": "dimensionless",
            "sta_code_child": "dimensionless",
            "sta_code_parent": "dimensionless",
            "sta_comment": "dimensionless",
            "sta_city": "dimensionless",
            "sta_crs": "dimensionless",
            "sta_epsg": "dimensionless",
            "sta_kp": "m",
            "sta_altitude_staff_gauge": "mm",
            "sta_date_altitude_ref": "dimensionless",
            "sta_date_start": "dimensionless",
            "sta_date_end": "dimensionless",
            "sta_publication_right": "dimensionless",
            "sta_time_data_gap": "min",
            "sta_time_discontinuity": "min",
            "sta_impact_local": "dimensionless",
            "sta_display_level": "dimensionless",
            "sta_dual_staff_gauge": "dimensionless",
            "sta_qual_lowflow": "dimensionless",
            "sta_qual_meanflow": "dimensionless",
            "sta_qual_highflow": "dimensionless",
            "sta_purpose": "dimensionless",
            "sta_comment_impact_local": "dimensionless",
            "sta_date_update": "dimensionless",
            "sit_code_h3": "dimensionless",
            "sta_main_prod_name": "dimensionless",
            "sta_main_prod_name_short": "dimensionless",
            "sta_main_prod_code": "dimensionless",
            "sta_x_l2e": "m",
            "sta_y_l2e": "m",
            "sta_x_l93": "m",
            "sta_y_l93": "m",
            "sta_x_w84": "degree E",
            "sta_y_w84": "degree N",
            "sta_x_l2e_snap": "m",
            "sta_y_l2e_snap": "m",
            "sta_x_l93_snap": "m",
            "sta_y_l93_snap": "m",
            "sta_x_w84_snap": "degree E",
            "sta_y_w84_snap": "degree N",
            "sta_area_snap": "km^2",
            "sta_altitude_snap": "m.a.s.l.",
            "sta_territory": "dimensionless",
        }

        # Assign units to the variables in the Dataset
        for var_name in units_dict:
            if var_name in ds_from_df.data_vars:
                ds_from_df[var_name].attrs["units"] = units_dict[var_name]

        return ds_from_df

    def cache_forcing_xrdataset(self):
        """Save all basin-forcing data in a netcdf file in the cache directory.

        """
        cache_npy_file = CACHE_DIR.joinpath("camels_fr_forcing.npy")
        json_file = CACHE_DIR.joinpath("camels_fr_forcing.json")
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

        units = ["mm/day", "-", "°C", "mm/day", "mm/day", "mm/day", "m/s", "g/kg", "J/cm^2", "J/cm^2", "-", "-", "mm/day", "°C", "°C"]
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
            attrs={"forcing_type": ""},
        )

    def cache_streamflow_xrdataset(self):
        """Save all basins' streamflow data in a netcdf file in the cache directory

        """
        cache_npy_file = CACHE_DIR.joinpath("camels_fr_streamflow.npy")
        json_file = CACHE_DIR.joinpath("camels_fr_streamflow.json")
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
