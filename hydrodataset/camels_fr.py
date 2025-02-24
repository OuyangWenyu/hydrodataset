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
        # attr  todo: there are two attribution folders, how to deal?
        attr_dir = camels_db.joinpath(
            "CAMELS_FR_attributes",
            "static_attributes",
            # "time_series_statistics"
        )
        attr_key_lst = [    # todo: the commented attribution files have different number of rows with station number
            "geology",
            "human_influences_dams",
            "hydrogeology",
            "land_cover",
            # "site_general",   # metadata   "sit_area_hydro", hydrological catchment area
            # "soil_general",
            # "soil_quantiles",
            # "station_general",  # metadata
            "topography_general",
            # "topography_quantiles",
        ]
        gauge_id_file = attr_dir.joinpath("CAMELS_FR_geology_attributes.csv")

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
            the time range, for example, ["1970-01-01", "2021-12-31"]
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
            the time range, for example, ["1970-01-01", "2021-12-31"]
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
        # other units are m3/s -> ft3/s
        y = y * 35.314666721489
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
            the time range, for example, ["1970-01-01", "2021-12-31"]
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
        data_folder = self.data_source_description["CAMELS_ATTR_DIR"]
        key_lst = self.data_source_description["CAMELS_ATTR_KEY_LST"]
        f_dict = {}
        var_dict = {}
        var_lst = []
        out_lst = []
        gage_dict = self.sites
        camels_str = "CAMELS_FR_"
        sep_ = ";"
        for key in key_lst:
            data_file = os.path.join(data_folder, camels_str + key + "_attributes.csv")
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
        return self.read_constant_cols(gage_id_lst, ["sit_area_hydro"], is_return_dict=False)   # todo:

    def read_mean_prcp(self, gage_id_lst, unit="mm/d") -> xr.Dataset:  # todo:
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
        data = self.read_constant_cols(
            gage_id_lst,
            ["cli_prec_mean"],
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
        cache_npy_file = CACHE_DIR.joinpath("camels_fr_forcing.npy")
        json_file = CACHE_DIR.joinpath("camels_fr_forcing.json")
        variables = self.get_relevant_cols()
        basins = self.sites["sta_code_h3"].values
        t_range = ["1970-01-01", "2021-12-31"]
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
        t_range = ["1970-01-01", "2021-12-31"]
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

        attr_files = self.data_source_dir.glob("CAMELS_FR_*.csv")
        attrs = {
            f.stem.split("_")[1]: pd.read_csv(
                f, sep=",", index_col=0, dtype={"sta_code_h3": str}
            )
            for f in attr_files
        }

        # attrs_df = pd.concat(attrs.values(), axis=1)
        attrs_df = attrs

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
            "geo_dom_class": "dimensionless",
            "geo_su": "%",
            "geo_ss": "%",
            "geo_py": "%",
            "geo_sm": "%",
            "geo_sc": "%",
            "geo_ev": "%",
            "geo_va": "%",
            "geo_vi": "%",
            "geo_vb": "%",
            "geo_pa": "%",
            "geo_pi": "%",
            "geo_pb": "%",
            "geo_mt": "%",
            "geo_wb": "%",
            "geo_ig": "%",
            "geo_nd": "%",
            "dam_n": "dimensionless",
            "dam_volume": "Mm^3",
            "dam_influence": "mm",
            "hgl_krs_not_karstic": "%",
            "hgl_krs_karstic": "%",
            "hgl_krs_unknown": "%",
            "hgl_thm_alluvial": "%",
            "hgl_thm_sedimentary": "%",
            "hgl_thm_bedrock": "%",
            "hgl_thm_intense_folded": "%",
            "hgl_thm_volcanism": "%",
            "hgl_thm_unknown": "%",
            "hgl_permeability": "log10(m^2)",
            "hgl_porosity": "dimensionless",
            "clc_2018_lvl1_dom_class": "dimensionless",
            "clc_2018_lvl1_1": "%",
            "clc_2018_lvl1_2": "%",
            "clc_2018_lvl1_3": "%",
            "clc_2018_lvl1_4": "%",
            "clc_2018_lvl1_5": "%",
            "clc_2018_lvl1_na": "%",
            "clc_2018_lvl2_dom_class": "dimensionless",
            "clc_2018_lvl2_11": "%",
            "clc_2018_lvl2_12": "%",
            "clc_2018_lvl2_13": "%",
            "clc_2018_lvl2_14": "%",
            "clc_2018_lvl2_21": "%",
            "clc_2018_lvl2_22": "%",
            "clc_2018_lvl2_23": "%",
            "clc_2018_lvl2_24": "%",
            "clc_2018_lvl2_31": "%",
            "clc_2018_lvl2_32": "%",
            "clc_2018_lvl2_33": "%",
            "clc_2018_lvl2_41": "%",
            "clc_2018_lvl2_42": "%",
            "clc_2018_lvl2_51": "%",
            "clc_2018_lvl2_52": "%",
            "clc_2018_lvl2_na": "%",
            "clc_2018_lvl3_dom_class": "dimensionless",
            "clc_2018_lvl3_111": "%",
            "clc_2018_lvl3_112": "%",
            "clc_2018_lvl3_121": "%",
            "clc_2018_lvl3_122": "%",
            "clc_2018_lvl3_123": "%",
            "clc_2018_lvl3_124": "%",
            "clc_2018_lvl3_131": "%",
            "clc_2018_lvl3_132": "%",
            "clc_2018_lvl3_133": "%",
            "clc_2018_lvl3_141": "%",
            "clc_2018_lvl3_142": "%",
            "clc_2018_lvl3_211": "%",
            "clc_2018_lvl3_212": "%",
            "clc_2018_lvl3_213": "%",
            "clc_2018_lvl3_221": "%",
            "clc_2018_lvl3_222": "%",
            "clc_2018_lvl3_223": "%",
            "clc_2018_lvl3_231": "%",
            "clc_2018_lvl3_241": "%",
            "clc_2018_lvl3_242": "%",
            "clc_2018_lvl3_243": "%",
            "clc_2018_lvl3_244": "%",
            "clc_2018_lvl3_311": "%",
            "clc_2018_lvl3_312": "%",
            "clc_2018_lvl3_313": "%",
            "clc_2018_lvl3_321": "%",
            "clc_2018_lvl3_322": "%",
            "clc_2018_lvl3_323": "%",
            "clc_2018_lvl3_324": "%",
            "clc_2018_lvl3_331": "%",
            "clc_2018_lvl3_332": "%",
            "clc_2018_lvl3_333": "%",
            "clc_2018_lvl3_334": "%",
            "clc_2018_lvl3_335": "%",
            "clc_2018_lvl3_411": "%",
            "clc_2018_lvl3_412": "%",
            "clc_2018_lvl3_421": "%",
            "clc_2018_lvl3_422": "%",
            "clc_2018_lvl3_423": "%",
            "clc_2018_lvl3_511": "%",
            "clc_2018_lvl3_512": "%",
            "clc_2018_lvl3_521": "%",
            "clc_2018_lvl3_522": "%",
            "clc_2018_lvl3_523": "%",
            "clc_2018_lvl3_na": "%",
            "clc_1990_lvl1_dom_class": "dimensionless",
            "clc_1990_lvl1_1": "%",
            "clc_1990_lvl1_2": "%",
            "clc_1990_lvl1_3": "%",
            "clc_1990_lvl1_4": "%",
            "clc_1990_lvl1_5": "%",
            "clc_1990_lvl1_na": "%",
            "clc_1990_lvl2_dom_class": "dimensionless",
            "clc_1990_lvl2_11": "%",
            "clc_1990_lvl2_12": "%",
            "clc_1990_lvl2_13": "%",
            "clc_1990_lvl2_14": "%",
            "clc_1990_lvl2_21": "%",
            "clc_1990_lvl2_22": "%",
            "clc_1990_lvl2_23": "%",
            "clc_1990_lvl2_24": "%",
            "clc_1990_lvl2_31": "%",
            "clc_1990_lvl2_32": "%",
            "clc_1990_lvl2_33": "%",
            "clc_1990_lvl2_41": "%",
            "clc_1990_lvl2_42": "%",
            "clc_1990_lvl2_51": "%",
            "clc_1990_lvl2_52": "%",
            "clc_1990_lvl2_na": "%",
            "clc_1990_lvl3_dom_class": "dimensionless",
            "clc_1990_lvl3_111": "%",
            "clc_1990_lvl3_112": "%",
            "clc_1990_lvl3_121": "%",
            "clc_1990_lvl3_122": "%",
            "clc_1990_lvl3_123": "%",
            "clc_1990_lvl3_124": "%",
            "clc_1990_lvl3_131": "%",
            "clc_1990_lvl3_132": "%",
            "clc_1990_lvl3_133": "%",
            "clc_1990_lvl3_141": "%",
            "clc_1990_lvl3_142": "%",
            "clc_1990_lvl3_211": "%",
            "clc_1990_lvl3_212": "%",
            "clc_1990_lvl3_213": "%",
            "clc_1990_lvl3_221": "%",
            "clc_1990_lvl3_222": "%",
            "clc_1990_lvl3_223": "%",
            "clc_1990_lvl3_231": "%",
            "clc_1990_lvl3_241": "%",
            "clc_1990_lvl3_242": "%",
            "clc_1990_lvl3_243": "%",
            "clc_1990_lvl3_244": "%",
            "clc_1990_lvl3_311": "%",
            "clc_1990_lvl3_312": "%",
            "clc_1990_lvl3_313": "%",
            "clc_1990_lvl3_321": "%",
            "clc_1990_lvl3_322": "%",
            "clc_1990_lvl3_323": "%",
            "clc_1990_lvl3_324": "%",
            "clc_1990_lvl3_331": "%",
            "clc_1990_lvl3_332": "%",
            "clc_1990_lvl3_333": "%",
            "clc_1990_lvl3_334": "%",
            "clc_1990_lvl3_335": "%",
            "clc_1990_lvl3_411": "%",
            "clc_1990_lvl3_412": "%",
            "clc_1990_lvl3_421": "%",
            "clc_1990_lvl3_422": "%",
            "clc_1990_lvl3_423": "%",
            "clc_1990_lvl3_511": "%",
            "clc_1990_lvl3_512": "%",
            "clc_1990_lvl3_521": "%",
            "clc_1990_lvl3_522": "%",
            "clc_1990_lvl3_523": "%",
            "clc_1990_lvl3_na": "%",
            "top_altitude_mean": "m.a.s.l.",
            "top_slo_mean": "degree",
            "top_dist_outlet_mean": "km",
            "top_itopo_mean": "dimensionless",
            "top_slo_ori_n": "%",
            "top_slo_ori_ne": "%",
            "top_slo_ori_e": "%",
            "top_slo_ori_se": "%",
            "top_slo_ori_s": "%",
            "top_slo_ori_sw": "%",
            "top_slo_ori_w": "%",
            "top_slo_ori_nw": "%",
            "top_drainage_density": "km km^2",
            "top_mor_form_factor_horton": "dimensionless",
            "top_mor_form_factor_square": "dimensionless",
            "top_mor_shape_factor": "dimensionless",
            "top_mor_compact_coef": "dimensionless",
            "top_mor_circ_ratio": "dimensionless",
            "top_mor_elong_ratio_circ": "dimensionless",
            "top_mor_elong_ratio_catchment": "dimensionless",
            "top_mor_relief_ratio": "dimensionless",
            "top_slo_flat": "%",
            "top_slo_gentle": "%",
            "top_slo_moderate": "%",
            "top_slo_strong": "%",
            "top_slo_steep": "%",
            "top_slo_very_steep": "%",
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
            "hym_q_na_period": "%",
            "hym_q_na_total": "%",
            "hym_q_n_year": "dimensionless",
            "hym_q_questionable": "%",
            "hym_q_unqualified": "%",
            "hym_q_anomaly_inrae": "%",
            "hym_q_low_uncertainty_inrae": "dimensionless",
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

        units = ["L/s", "mm/day", "-", "-", "-", "-", "-", "mm/day", "-", "°C", "mm/day", "mm/day", "mm/day", "m/s", "g/kg", "J/cm^2", "J/cm^2", "-", "-", "mm/day", "°C", "°C"]
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

    def cache_xrdataset(self):
        """
        Save all data in a netcdf file in the cache directory

        """

        warnings.warn("Check you units of all variables")
        ds_attr = self.cache_attributes_xrdataset()
        ds_attr.to_netcdf(CACHE_DIR.joinpath("camelsfr_attributes.nc"))
        ds_streamflow = self.cache_streamflow_xrdataset()
        ds_forcing = self.cache_forcing_xrdataset()
        ds = xr.merge([ds_streamflow, ds_forcing])
        ds.to_netcdf(CACHE_DIR.joinpath("camelsfr_timeseries.nc"))
