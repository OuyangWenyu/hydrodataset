import os
import collections
import pandas as pd
import numpy as np
from typing import Union
from pandas.api.types import is_string_dtype, is_numeric_dtype
from tqdm import tqdm
import xarray as xr
from hydroutils import hydro_time
from hydrodataset import CACHE_DIR, CAMELS_REGIONS
from hydrodataset.camels import Camels
import json

CAMELS_NO_DATASET_ERROR_LOG = (
    "We cannot read this dataset now. Please check if you choose correctly:\n"
    + str(CAMELS_REGIONS)
)

camelsaus_arg = {
    "forcing_type": "observation",
    "gauge_id_tag": "station_id",
    "area_tag": ["catchment_area", ],
    "meanprcp_unit_tag": [["p_mean"], "mm/d"],
    "time_range": {
        "observation": ["1990-01-01", "2010-01-01"],
    },
    "b_nestedness": False,
}

class CamelsAus(Camels):
    def __init__(
        self,
        data_path=os.path.join("camels", "camels_aus"),
        download=False,
        region: str = "AUS",
        arg: dict = camelsaus_arg,
    ):
        """
        Initialization for CAMELS-AUS dataset

        Parameters
        ----------
        data_path
            where we put the dataset.
            we already set the ROOT directory for hydrodataset,
            so here just set it as a relative path,
            by default "camels/camels_aus"
        download
            if true, download, by default False
        region
            the default is CAMELS-AUS
        """
        super().__init__(data_path, download, region, arg)

    def set_data_source_describe(self) -> collections.OrderedDict:
        """
        the files in the dataset and their location in file system

        Returns
        -------
        collections.OrderedDict
            the description for a CAMELS-AUS dataset
        """
        camels_db = self.data_source_dir

        if self.region == "AUS":
            return self._set_data_source_camelsaus_describe(camels_db)
        elif self.region == "AUS_v2":
            return self._set_data_source_camelsausv2_describe(camels_db)
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)

    def _set_data_source_camelsausv2_describe(self, camels_db):
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
            "CAMELS_AUS_v2_BasinOutlets_adopted.shp",
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
        nestedness_information_file = None
        base_url = "https://download.pangaea.de/dataset/921850"
        download_url_lst = [
            f"{base_url}/files/01_id_name_metadata.zip",
            f"{base_url}/files/02_location_boundary_area.zip",
            f"{base_url}/files/03_streamflow.zip",
            f"{base_url}/file/04_attributes.zip",
            f"{base_url}/file/05_hydrometeorology.zip",
            f"{base_url}/file/CAMELS_AUS_Attributes&Indices_MasterTable.csv",
            f"{base_url}/file/CAMELS_AUS_ReferenceList.pdf",
            f"{base_url}/file/Units_01_TimeseriesData.pdf",
            f"{base_url}/file/Units_02_AttributeMasterTable.pdf",
        ]

        return collections.OrderedDict(
            CAMELS_DIR=camels_db,
            CAMELS_FLOW_DIR=flow_dir,
            CAMELS_FORCING_DIR=forcing_dir,
            CAMELS_ATTR_DIR=attr_dir,
            CAMELS_GAUGE_FILE=gauge_id_file,
            CAMELS_NESTEDNESS_FILE=nestedness_information_file,
            CAMELS_BASINS_SHP_FILE=camels_shp_file,
            CAMELS_DOWNLOAD_URL_LST=download_url_lst,
        )

    def read_site_info(self) -> pd.DataFrame:
        """
        Read the basic information of gages in a CAMELS-AUS dataset

        Returns
        -------
        pd.DataFrame
            basic info of gages
        """
        camels_file = self.data_source_description["CAMELS_GAUGE_FILE"]
        if self.region in ["AUS", "AUS_v2"]:
            data = pd.read_csv(camels_file, sep=",", dtype={self.gauge_id_tag: str})
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)
        return data

    def get_constant_cols(self) -> np.ndarray:
        """
        all readable attrs in CAMELS-AUS

        Returns
        -------
        np.array
            attribute types
        """
        attr_all_file = os.path.join(
            self.data_source_description["CAMELS_DIR"],
            "CAMELS_AUS_Attributes-Indices_MasterTable.csv",
        )
        camels_aus_attr_indices_data = pd.read_csv(attr_all_file, sep=",")
        # exclude station id
        return camels_aus_attr_indices_data.columns.values[1:]

    def get_relevant_cols(self) -> np.ndarray:
        """
        all readable forcing types in CAMELS-AUS

        "precipitation_AWAP",
        "precipitation_SILO",
        "precipitation_var_AWAP",
        "et_morton_actual_SILO",
        "et_morton_point_SILO",
        "et_morton_wet_SILO",
        "et_short_crop_SILO",
        "et_tall_crop_SILO",
        "evap_morton_lake_SILO",
        "evap_pan_SILO",
        "evap_syn_SILO",
        "solarrad_AWAP",
        "tmax_AWAP",
        "tmin_AWAP",
        "vprp_AWAP",
        "mslp_SILO",
        "radiation_SILO",
        "rh_tmax_SILO",
        "rh_tmin_SILO",
        "tmax_SILO",
        "tmin_SILO",
        "vp_deficit_SILO",
        "vp_SILO",

        Returns
        -------
        np.array
            forcing types
        """
        if self.region == "AUS":
            forcing_types = []
            for root, dirs, files in os.walk(
                self.data_source_description["CAMELS_FORCING_DIR"]
            ):
                if root == self.data_source_description["CAMELS_FORCING_DIR"]:
                    continue
                forcing_types.extend(
                    file[:-4] for file in files if file != "ClimaticIndices.csv"
                )
            the_cols = np.array(forcing_types)
        elif self.region == "AUS_v2":
            forcing_types = []
            for root, dirs, files in os.walk(
                self.data_source_description["CAMELS_FORCING_DIR"]
            ):
                if root == self.data_source_description["CAMELS_FORCING_DIR"]:
                    continue
                forcing_types.extend(
                    file[:-4]
                    for file in files
                    if file not in ["ClimaticIndices.csv", "desktop.ini"]
                )
            the_cols = np.array(forcing_types)
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)
        return the_cols

    def get_target_cols(self) -> np.ndarray:
        """
        For CAMELS-AUS, the target vars are streamflows

        QualityCodes are not streamflow data.
        MLd means "1 Megaliters Per Day"; 1 MLd = 0.011574074074074 cubic-meters-per-second
        mmd means "mm/day"

        Returns
        -------
        np.array
            streamflow types
        """
        return np.array(
            [
                "streamflow_MLd",
                "streamflow_MLd_inclInfilled",
                "streamflow_mmd",
            ]
        )

    def read_target_cols(
        self,
        gage_id_lst: Union[list, np.array] = None,
        t_range: list = None,
        target_cols: Union[list, np.array] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        read target values; for CAMELS-AUS, they are streamflows

        default target_cols is an one-value list
        Notice: the unit of target outputs in different regions are not totally same

        Parameters
        ----------
        gage_id_lst
            station ids
        t_range
            the time range, for example, ["1990-01-01", "2000-01-01"]
        target_cols
            the default is None, but we need at least one default target.
            For CAMELS-AUS, it's ["streamflow_mmd"]
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
            [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
            chosen_data = flow_data[gage_id_lst].values[ind1, :]
            chosen_data[chosen_data < 0] = np.nan
            y[:, ind2, k] = chosen_data.T
            # ML/d-->m3/s
            if target_cols[k] == "streamflow_MLd":
                y = y / 84.6
        # Keep unit of streamflow unified: we use ft3/s here
        # other units are m3/s -> ft3/s
        y = self.unit_convert_streamflow_m3tofoot3(y)
        return y

    def read_relevant_cols(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        forcing_type="AWAP",
        **kwargs,
    ) -> np.ndarray:
        """
        Read forcing data

        Parameters
        ----------
        gage_id_lst
            station ids
        t_range
            the time range, for example, ["1990-01-01", "2010-01-01"]
        var_lst
            forcing variable types, "precipitation_AWAP", "precipitation_SILO", "precipitation_var_AWAP", "et_morton_actual_SILO", "et_morton_point_SILO",
                   "et_morton_wet_SILO", "et_short_crop_SILO", "et_tall_crop_SILO", "evap_morton_lake_SILO", "evap_pan_SILO", "evap_syn_SILO",
                   "solarrad_AWAP", "tmax_AWAP", "tmin_AWAP", "vprp_AWAP", "mslp_SILO", "radiation_SILO", "rh_tmax_SILO", "rh_tmin_SILO", "tmax_SILO",
                   "tmin_SILO", "vp_deficit_SILO", "vp_SILO",
        forcing_type
            now only for CAMELS-AUS, there are two types: AWAP, SILO
        Returns
        -------
        np.array
            forcing data
        """
        t_range_list = hydro_time.t_range_days(t_range)
        nt = t_range_list.shape[0]
        x = np.full([len(gage_id_lst), nt, len(var_lst)], np.nan)
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
            forcing_data = pd.read_csv(os.path.join(forcing_dir, var_lst[k] + ".csv"))
            df_date = forcing_data[["year", "month", "day"]]
            date = pd.to_datetime(df_date).values.astype("datetime64[D]")
            [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
            chosen_data = forcing_data[gage_id_lst].values[ind1, :]
            x[:, ind2, k] = chosen_data.T
        return x

    def read_attr_all_in_one_file(self):
        """
        Read all attr data in CAMELS_AUS

        Returns
        -------
        np.array
            all attr data in CAMELS_AUS
        """
        attr_all_file = os.path.join(
            self.data_source_description["CAMELS_DIR"],
            "CAMELS_AUS_Attributes-Indices_MasterTable.csv",
        )
        all_attr = pd.read_csv(attr_all_file, sep=",")
        var_lst = self.get_constant_cols().tolist()
        data_temp = all_attr[var_lst]
        # for factorized data, we need factorize all gages' data to keep the factorized number same all the time
        out = np.full([self.n_gage, len(var_lst)], np.nan)
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
        id_lst_all = self.gage
        # Notice the sequence of station ids ! Some id_lst_all are not sorted, so don't use np.intersect1d
        ind_grid = [id_lst_all.index(tmp) for tmp in gage_id_lst]
        temp = attr_all[ind_grid, :]
        out = temp[:, ind_var]
        return (out, var_dict, f_dict) if is_return_dict else out

    def cache_forcing_np_json(self):
        """
        Save all basin-forcing data in a numpy array file in the cache directory.

        Because it takes much time to read data from csv files,
        it is a good way to cache data as a numpy file to speed up the reading.
        In addition, we need a document to explain the meaning of all dimensions.

        """
        cache_npy_file = CACHE_DIR.joinpath("camels_aus_forcing.npy")
        json_file = CACHE_DIR.joinpath("camels_aus_forcing.json")
        variables = self.get_relevant_cols()
        basins = self.gage
        t_range = self.time_range
        times = [
            hydro_time.t2str(tmp)
            for tmp in hydro_time.t_range_days(t_range).tolist()
        ]
        data_info = collections.OrderedDict(
            {
                "dim": ["basin", "time", "variable"],
                "basin": basins,
                "time": times,
                "variable": variables.tolist(),
            }
        )
        with open(json_file, "w") as FP:
            json.dump(data_info, FP, indent=4)
        data = self.read_relevant_cols(
            gage_id_lst=basins,
            t_range=t_range,
            var_lst=variables.tolist(),
        )
        np.save(cache_npy_file, data)

    def cache_streamflow_np_json(self):
        """
        Save all basins' streamflow data in a numpy array file in the cache directory
        """
        cache_npy_file = CACHE_DIR.joinpath("camels_aus_streamflow.npy")
        json_file = CACHE_DIR.joinpath("camels_aus_streamflow.json")
        variables = self.get_target_cols()
        basins = self.gage
        t_range = self.time_range
        times = [
            hydro_time.t2str(tmp) for tmp in hydro_time.t_range_days(t_range).tolist()
        ]
        data_info = collections.OrderedDict(
            {
                "dim": ["basin", "time", "variable"],
                "basin": basins,
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

        attr_all, var_lst_all, var_dict, f_dict = self.read_attr_all_in_one_file()
        basins = self.gage
        attrs_df = pd.DataFrame(data=attr_all[0:, 0:], index=basins, columns=var_lst_all)

        # unify id to basin
        attrs_df.index.name = "basin"
        # We use xarray dataset to cache all data
        ds_from_df = attrs_df.to_xarray()
        units_dict = {
            "station_id": "dimensionless",
            "station_name": "dimensionless",
            "drainage_division": "dimensionless",
            "river_region": "dimensionless",
            "notes": "dimensionless",
            "lat_outlet": "degree",
            "long_outlet": "degree",
            "lat_centroid": "degree",
            "long_centroid": "degree",
            "map_zone": "dimensionless",
            "catchment_area": "km^2",
            "state_outlet": "dimensionless",
            "state_alt": "dimensionless",
            "daystart": "dimensionless",
            "daystart_P": "dimensionless",
            "daystart_Q": "dimensionless",
            "nested_status": "dimensionless",
            "next_station_ds": "dimensionless",
            "num_nested_within": "dimensionless",
            "start_date": "dimensionless",
            "end_date": "dimensionless",
            "prop_missing_data": "percent",
            "q_uncert_num_curves": "dimensionless",
            "q_uncert_n": "dimensionless",
            "q_uncert_q10": "mm/d",
            "q_uncert_q10_upper": "percent",
            "q_uncert_q10_lower": "percent",
            "q_uncert_q50": "mm/d",
            "q_uncert_q50_upper": "percent",
            "q_uncert_q50_lower": "percent",
            "q_uncert_q90": "mm/d",
            "q_uncert_q90_upper": "percent",
            "q_uncert_q90_lower": "percent",
            "p_mean": "mm/d",
            "pet_mean": "mm/d",
            "aridity": "dimensionless",
            "p_seasonality": "dimensionless",
            "frac_snow": "dimensionless",
            "high_prec_freq": "d/y",
            "high_prec_dur": "days",
            "high_prec_timing": "season",
            "low_prec_freq": "d/y",
            "low_prec_dur": "days",
            "low_prec_timing": "season",
            "q_mean": "mm/d",
            "runoff_ratio": "dimensionless",
            "stream_elas": "dimensionless",
            "slope_fdc": "dimensionless",
            "baseflow_index": "dimensionless",
            "hdf_mean": "day of year",
            "Q5": "mm/d",
            "Q95": "mm/d",
            "high_q_freq": "d/y",
            "high_q_dur": "days",
            "low_q_freq": "d/y",
            "low_q_dur": "days",
            "zero_q_freq": "d/y",
            "geol_prim": "dimensionless",
            "geol_prim_prop": "dimensionless",
            "geol_sec": "dimensionless",
            "geol_sec_prop": "dimensionless",
            "unconsoldted": "dimensionless",
            "igneous": "dimensionless",
            "silicsed": "dimensionless",
            "carbnatesed": "dimensionless",
            "othersed": "dimensionless",
            "metamorph": "dimensionless",
            "sedvolc": "dimensionless",
            "oldrock": "dimensionless",
            "claya": "percent",
            "clayb": "percent",
            "sanda": "percent",
            "solum_thickness": "m",
            "ksat": "mm/h",
            "solpawhc": "mm",
            "elev_min": "m",
            "elev_max": "m",
            "elev_mean": "m",
            "elev_range": "m",
            "mean_slope_pct": "percent",
            "upsdist": "km",
            "strdensity": "1/km",
            "strahler": "dimensionless",
            "elongratio": "dimensionless",
            "relief": "dimensionless",
            "reliefratio": "dimensionless",
            "mrvbf_prop_0": "dimensionless",
            "mrvbf_prop_1": "dimensionless",
            "mrvbf_prop_2": "dimensionless",
            "mrvbf_prop_3": "dimensionless",
            "mrvbf_prop_4": "dimensionless",
            "mrvbf_prop_5": "dimensionless",
            "mrvbf_prop_6": "dimensionless",
            "mrvbf_prop_7": "dimensionless",
            "mrvbf_prop_8": "dimensionless",
            "mrvbf_prop_9": "dimensionless",
            "confinement": "dimensionless",
            "lc01_extracti": "dimensionless",
            "lc03_waterbo": "dimensionless",
            "lc04_saltlak": "dimensionless",
            "lc05_irrcrop": "dimensionless",
            "lc06_irrpast": "dimensionless",
            "lc07_irrsuga": "dimensionless",
            "lc08_rfcropp": "dimensionless",
            "lc09_rfpastu": "dimensionless",
            "lc10_rfsugar": "dimensionless",
            "lc11_wetlands": "dimensionless",
            "lc14_tussclo": "dimensionless",
            "lc15_alpineg": "dimensionless",
            "lc16_openhum": "dimensionless",
            "lc18_opentus": "dimensionless",
            "lc19_shrbsca": "dimensionless",
            "lc24_shrbden": "dimensionless",
            "lc25_shrbope": "dimensionless",
            "lc31_forclos": "dimensionless",
            "lc32_foropen": "dimensionless",
            "lc33_woodope": "dimensionless",
            "lc34_woodspa": "dimensionless",
            "lc35_urbanar": "dimensionless",
            "prop_forested": "dimensionless",
            "nvis_grasses_n": "dimensionless",
            "nvis_grasses_e": "dimensionless",
            "nvis_forests_n": "dimensionless",
            "nvis_forests_e": "dimensionless",
            "nvis_shrubs_n": "dimensionless",
            "nvis_shrubs_e": "dimensionless",
            "nvis_woodlands_n": "dimensionless",
            "nvis_woodlands_e": "dimensionless",
            "nvis_bare_n": "dimensionless",
            "nvis_bare_e": "dimensionless",
            "nvis_nodata_n": "dimensionless",
            "nvis_nodata_e": "dimensionless",
            "distupdamw": "km",
            "impound_fac": "dimensionless",
            "flow_div_fac": "dimensionless",
            "leveebank_fac": "dimensionless",
            "infrastruc_fac": "dimensionless",
            "settlement_fac": "dimensionless",
            "extract_ind_fac": "dimensionless",
            "landuse_fac": "dimensionless",
            "catchment_di": "dimensionless",
            "flow_regime_di": "dimensionless",
            "river_di": "dimensionless",
            "pop_mean": "km^2",
            "pop_max": "km^2",
            "pop_gt_1": "dimensionless",
            "pop_gt_10": "dimensionless",
            "erosivity": "MJ mm ha-1 h-1",
            "anngro_mega": "dimensionless",
            "anngro_meso": "dimensionless",
            "anngro_micro": "dimensionless",
            "gromega_seas": "dimensionless",
            "gromeso_seas": "dimensionless",
            "gromicro_seas": "dimensionless",
            "npp_ann": "tC Ha-1",
            "npp_1": "tC Ha-1",
            "npp_2": "tC Ha-1",
            "npp_3": "tC Ha-1",
            "npp_4": "tC Ha-1",
            "npp_5": "tC Ha-1",
            "npp_6": "tC Ha-1",
            "npp_7": "tC Ha-1",
            "npp_8": "tC Ha-1",
            "npp_9": "tC Ha-1",
            "npp_10": "tC Ha-1",
            "npp_11": "tC Ha-1",
            "npp_12": "tC Ha-1",
        }

        # Assign units to the variables in the Dataset
        for var_name in units_dict:
            if var_name in ds_from_df.data_vars:
                ds_from_df[var_name].attrs["units"] = units_dict[var_name]

        return ds_from_df

    def cache_forcing_xrdataset(self):
        """Save all basin-forcing data in a netcdf file in the cache directory.

        """
        cache_npy_file = CACHE_DIR.joinpath("camels_aus_forcing.npy")
        json_file = CACHE_DIR.joinpath("camels_aus_forcing.json")
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

        units = ["mm/d", "mm/d", "mm^2 /d^2", "mm/d", "mm/d", "mm/d", "mm/d", "mm/d", "mm/d", "mm/d", "mm/d", "MJ/m^2",
                 "°C", "°C", "°C", "°C", "MJ/m^2", "%", "%", "°C", "°C", "hPa", "hPa",]
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
        cache_npy_file = CACHE_DIR.joinpath("camels_aus_streamflow.npy")
        json_file = CACHE_DIR.joinpath("camels_aus_streamflow.json")
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
