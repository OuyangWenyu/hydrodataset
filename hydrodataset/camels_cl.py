import fnmatch
import os
import collections
import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype, is_numeric_dtype
from typing import Union
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

camelscl_arg = {
    "forcing_type": "observation",
    "gauge_id_tag": "gauge_id",
    "area_tag": ["area",],
    "meanprcp_unit_tag": [["p_mean_cr2met"], "mm/d"],
    "time_range": {
        "observation": ["1995-01-01", "2015-01-01"],
    },
    "target_cols": ["streamflow_m3s", "streamflow_mm"],
    "b_nestedness": False,
    "forcing_unit": ["mm/day", "mm/day", "mm/day", "mm/day", "°C", "°C", "°C", "mm/day", "mm/day", "mm"],
    "data_file_attr": {
        "sep": "\t",
        "header": 0,
        # "attr_file_str": ["CAMELS_DK_", ".csv", ]
    },
}

class CamelsCl(Camels):
    def __init__(
        self,
        data_path=os.path.join("camels", "camels_cl"),
        download=False,
        region: str = "CL",
        arg: dict = camelscl_arg,
    ):
        """
        Initialization for CAMELS-CL dataset

        Parameters
        ----------
        data_path
            where we put the dataset.
            we already set the ROOT directory for hydrodataset,
            so here just set it as a relative path,
            by default "camels/camels_cl"
        download
            if true, download, by default False
        region
            the default is CAMELS-CL
        """
        super().__init__(data_path, download, region, arg)

    def _set_data_source_camels_describe(self, camels_db):
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
        nestedness_information_file = None
        base_url = "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018"
        download_url_lst = [
            f"{base_url}/1_CAMELScl_attributes.zip",
            f"{base_url}/2_CAMELScl_streamflow_m3s.zip",
            f"{base_url}/3_CAMELScl_streamflow_mm.zip",
            f"{base_url}/4_CAMELScl_precip_cr2met.zip",
            f"{base_url}/5_CAMELScl_precip_chirps.zip",
            f"{base_url}/6_CAMELScl_precip_mswep.zip",
            f"{base_url}/7_CAMELScl_precip_tmpa.zip",
            f"{base_url}/8_CAMELScl_tmin_cr2met.zip",
            f"{base_url}/9_CAMELScl_tmax_cr2met.zip",
            f"{base_url}/10_CAMELScl_tmean_cr2met.zip",
            f"{base_url}/11_CAMELScl_pet_8d_modis.zip",
            f"{base_url}/12_CAMELScl_pet_hargreaves.zip",
            f"{base_url}/13_CAMELScl_swe.zip",
            f"{base_url}/14_CAMELScl_catch_hierarchy.zip",
            f"{base_url}/CAMELScl_catchment_boundaries.zip",
        ]

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
            CAMELS_NESTEDNESS_FILE=nestedness_information_file,
            CAMELS_BASINS_SHP_FILE=camels_shp_file,
            CAMELS_DOWNLOAD_URL_LST=download_url_lst,
        )

    def read_site_info(self) -> pd.DataFrame:
        """
        Read the basic information of gages in a CAMELS-CL dataset

        Returns
        -------
        pd.DataFrame
            basic info of gages
        """
        camels_file = self.data_source_description["CAMELS_GAUGE_FILE"]
        return pd.read_csv(camels_file, sep="\t", index_col=0)

    def get_constant_cols(self) -> np.ndarray:
        """
        all readable attrs in CAMELS-CL

        Returns
        -------
        np.array
            attribute types
        """
        camels_cl_attr_data = self.sites
        # exclude station id
        return camels_cl_attr_data.index.values

    def get_relevant_cols(self) -> np.ndarray:
        """
        all readable forcing types in CAMELS-CL

        Returns
        -------
        np.array
            forcing types
        """
        return np.array(
            [
                "_".join(str(forcing_dir).split(os.sep)[-1].split("_")[2:])
                for forcing_dir in self.data_source_description["CAMELS_FORCING_DIR"]
            ]
        )

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
        station_ids = self.sites.columns.values
        # for 7-digit id, replace the space with 0 to get a 8-digit id
        cl_station_ids = [
            station_id.split(" ")[-1].zfill(8) for station_id in station_ids
        ]
        return np.array(cl_station_ids)

    def read_target_cols(
        self,
        gage_id_lst: Union[list, np.array] = None,
        t_range: list = None,
        target_cols: Union[list, np.array] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        read target values; for CAMELS-CL, they are streamflows

        default target_cols is an one-value list
        Notice: the unit of target outputs in different regions are not totally same

        Parameters
        ----------
        gage_id_lst
            station ids
        t_range
            the time range, for example, ["1995-01-01", "2015-01-01"]
        target_cols
            the default is None, but we need at least one default target.
            For CAMELS-CL, it's ["streamflow_m3s"]
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
            date = pd.to_datetime(flow_data.index.values).values.astype("datetime64[D]")
            [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
            station_ids = [id_.zfill(8) for id_ in flow_data.columns.values]
            assert all(x < y for x, y in zip(station_ids, station_ids[1:]))
            ind3 = [station_ids.index(tmp) for tmp in gage_id_lst]
            # to guarantee the sequence is not changed we don't use np.intersect1d
            chosen_data = flow_data.iloc[ind1, ind3].replace("\s+", np.nan, regex=True)
            chosen_data = chosen_data.astype(float)
            chosen_data[chosen_data < 0] = np.nan
            y[:, ind2, k] = chosen_data.values.T
        # Keep unit of streamflow unified: we use ft3/s here
        # other units are m3/s -> ft3/s
        y = self.unit_convert_streamflow_m3tofoot3(y)
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
            the time range, for example, ["1995-01-01", "2015-01-01"]
        var_lst
            forcing variable types, "precip_cr2met", "precip_chirps", "precip_mswep", "precip_tmpa", "tmin_cr2met", "tmax_cr2met", "tmean_cr2met", "pet_8d_modis", "pet_hargreaves", "swe",
        forcing_type
            support for CAMELS-CL, there are only one type: observation
        Returns
        -------
        np.array
            forcing data
        """
        t_range_list = hydro_time.t_range_days(t_range)
        nt = t_range_list.shape[0]
        x = np.full([len(gage_id_lst), nt, len(var_lst)], np.nan)
        for k in tqdm(range(len(var_lst)), desc="Read forcing data of CAMELS-CL"):
            for tmp in os.listdir(self.data_source_description["CAMELS_DIR"]):
                if fnmatch.fnmatch(tmp, "*" + var_lst[k]):
                    tmp_ = os.path.join(self.data_source_description["CAMELS_DIR"], tmp)
                    if os.path.isdir(tmp_):
                        forcing_file = os.path.join(tmp_, os.listdir(tmp_)[0])
            forcing_data = pd.read_csv(forcing_file, sep="\t", index_col=0)
            date = pd.to_datetime(forcing_data.index.values).values.astype(
                "datetime64[D]"
            )
            [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
            station_ids = [id_.zfill(8) for id_ in forcing_data.columns.values]
            assert all(x < y for x, y in zip(station_ids, station_ids[1:]))
            ind3 = [station_ids.index(tmp) for tmp in gage_id_lst]
            # to guarantee the sequence is not changed we don't use np.intersect1d
            chosen_data = forcing_data.iloc[ind1, ind3].replace(
                "\s+", np.nan, regex=True
            )
            x[:, ind2, k] = chosen_data.values.T
        return x

    def read_attr_all_in_one_file(self):
        """
        Read all attr data in CAMELS_CL

        Returns
        -------
        np.array
            all attr data in CAMELS_CL
        """
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
        # gage_all_attr = all_attr[all_attr['station_id'].isin(gage_id_lst)]
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

    def get_attribute_units_dict(self):
        """

        Returns
        -------

        """
        units_dict = {
            "gauge_name": "dimensionless",
            "gauge_lat": "degree south",
            "gauge_lon": "degree west",
            "record_period_start": "dimensionless",
            "record_period_end": "dimensionless",
            "n_obs": "day",
            "area": "km^2",
            "elev_gauge": "m.a.s.l.",
            "elev_mean": "m.a.s.l.",
            "elev_med": "m.a.s.l.",
            "elev_max": "m.a.s.l.",
            "elev_min": "m.a.s.l.",
            "slope_mean": "m/km",
            "nested_inner": "dimensionless",
            "nested_outer": "dimensionless",
            "location_type": "dimensionless",
            "geol_class_1st": "dimensionless",
            "geol_class_1st_frac": "dimensionless",
            "geol_class_2nd": "dimensionless",
            "geol_class_2nd_frac": "dimensionless",
            "carb_rocks_frac": "dimensionless",
            "crop_frac": "percent",
            "nf_frac": "percent",
            "fp_frac": "percent",
            "grass_frac": "percent",
            "shrub_frac": "percent",
            "wet_frac": "percent",
            "imp_frac": "percent",
            "lc_barren": "percent",
            "snow_frac": "percent",
            "lc_glacier": "percent",
            "fp_nf_index": "dimensionless",
            "forest_frac": "percent",
            "dom_land_cover": "dimensionless",
            "dom_land_cover_frac": "percent",
            "land_cover_missing": "percent",
            "p_mean_cr2met": "mm/day",
            "p_mean_chirps": "mm/day",
            "p_mean_mswep": "mm/day",
            "p_mean_tmpa": "mm/day",
            "pet_mean": "mm/day",
            "aridity_cr2met": "dimensionless",
            "aridity_chirps": "dimensionless",
            "aridity_mswep": "dimensionless",
            "aridity_tmpa": "dimensionless",
            "p_seasonality_cr2met": "dimensionless",
            "p_seasonality_chirps": "dimensionless",
            "p_seasonality_mswep": "dimensionless",
            "p_seasonality_tmpa": "dimensionless",
            "frac_snow_cr2met": "dimensionless",
            "frac_snow_chirps": "dimensionless",
            "frac_snow_mswep": "dimensionless",
            "frac_snow_tmpa": "dimensionless",
            "high_prec_freq_cr2met": "days/yr",
            "high_prec_freq_chirps": "days/yr",
            "high_prec_freq_mswep": "days/yr",
            "high_prec_freq_tmpa": "days/yr",
            "high_prec_dur_cr2met": "days",
            "high_prec_dur_chirps": "days",
            "high_prec_dur_mswep": "days",
            "high_prec_dur_tmpa": "days",
            "high_prec_timing_cr2met": "season",
            "high_prec_timing_chirps": "season",
            "high_prec_timing_mswep": "season",
            "high_prec_timing_tmpa": "season",
            "low_prec_freq_cr2met": "days/yr",
            "low_prec_freq_chirps": "days/yr",
            "low_prec_freq_mswep": "days/yr",
            "low_prec_freq_tmpa": "days/yr",
            "low_prec_dur_cr2met": "days",
            "low_prec_dur_chirps": "days",
            "low_prec_dur_mswep": "days",
            "low_prec_dur_tmpa": "days",
            "low_prec_timing_cr2met": "season",
            "low_prec_timing_chirps": "season",
            "low_prec_timing_mswep": "season",
            "low_prec_timing_tmpa": "season",
            "p_mean_spread": "dimensionless",
            "q_mean": "mm/day",
            "runoff_ratio_cr2met": "dimensionless",
            "runoff_ratio_chirps": "dimensionless",
            "runoff_ratio_mswep": "dimensionless",
            "runoff_ratio_tmpa": "dimensionless",
            "stream_elas_cr2met": "dimensionless",
            "stream_elas_chirps": "dimensionless",
            "stream_elas_mswep": "dimensionless",
            "stream_elas_tmpa": "dimensionless",
            "slope_fdc": "dimensionless",
            "baseflow_index": "dimensionless",
            "hfd_mean": "day of year",
            "Q95": "mm/day",
            "Q5": "mm/day",
            "high_q_freq": "days/yr",
            "high_q_dur": "days",
            "low_q_freq": "days/yr",
            "low_q_dur": "days",
            "zero_q_freq": "percent",
            "swe_ratio": "dimensionless",
            "sur_rights_n": "dimensionless",
            "sur_rights_flow": "m^3/s",
            "interv_degree": "dimensionless",
            "gw_rights_n": "dimensionless",
            "gw_rights_flow": "m^3/s",
            "big_dam": "dimensionless",
        }

        return units_dict

