import logging
import os
import collections
import pandas as pd
import numpy as np
from typing import Union
from tqdm import tqdm
import xarray as xr
from hydroutils import hydro_time
from hydrodataset import CACHE_DIR, CAMELS_REGIONS
from hydrodataset.camels import Camels, time_intersect_dynamic_data
from pandas.api.types import is_string_dtype, is_numeric_dtype
import json

CAMELS_NO_DATASET_ERROR_LOG = (
    "We cannot read this dataset now. Please check if you choose correctly:\n"
    + str(CAMELS_REGIONS)
)

camelsgb_arg = {
    "forcing_type": "observation",
    "gauge_id_tag": "gauge_id",
    "area_tag": ["area", ],
    "meanprcp_unit_tag": [["p_mean"], "mm/d"],
    "time_range": {
        "observation": ["1970-10-01", "2015-10-01"],
    },
    "target_cols": ["discharge_spec", "discharge_vol"],
    "b_nestedness": False,
    "forcing_unit": ["mm/day", "mm/day", "°C", "mm/day", "m^3/s", "mm/day", "g/kg", "W/m^2", "W/m^2", "m/s"],
    "data_file_attr": {
        "sep": ",",
        "header": 0,
        "attr_file_str": ["CAMELS_GB_", "_attributes.csv", ]
    },
}

class CamelsGb(Camels):
    def __init__(
        self,
        data_path=os.path.join("camels", "camels_gb"),
        download=False,
        region: str = "GB",
        arg: dict = camelsgb_arg,
    ):
        """
        Initialization for CAMELS-GB dataset

        Parameters
        ----------
        data_path
            where we put the dataset.
            we already set the ROOT directory for hydrodataset,
            so here just set it as a relative path,
            by default "camels/camels_gb"
        download
            if true, download, by default False
        region
            the default is CAMELS-GB
        """
        super().__init__(data_path, download, region, arg)

    def _set_data_source_camels_describe(self, camels_db):
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
        nestedness_information_file = None
        base_url = "https://data-package.ceh.ac.uk"
        download_url_lst = [
            f"{base_url}/data/8344e4f3-d2ea-44f5-8afa-86d2987543a9.zip",
        ]

        return collections.OrderedDict(
            CAMELS_DIR=camels_db,
            CAMELS_FLOW_DIR=flow_dir,
            CAMELS_FORCING_DIR=forcing_dir,
            CAMELS_ATTR_DIR=attr_dir,
            CAMELS_ATTR_KEY_LST=attr_key_lst,
            CAMELS_GAUGE_FILE=gauge_id_file,
            CAMELS_NESTEDNESS_FILE=nestedness_information_file,
            CAMELS_BASINS_SHP_FILE=camels_shp_file,
            CAMELS_DOWNLOAD_URL_LST=download_url_lst,
        )

    def get_constant_cols(self) -> np.ndarray:
        """
        all readable attrs in CAMELS-GB

        Returns
        -------
        np.array
            attribute types
        """
        data_folder = self.data_source_description["CAMELS_ATTR_DIR"]
        return self._get_constant_cols_some(
            data_folder, "CAMELS_GB_", "_attributes.csv", ","
        )

    def get_relevant_cols(self) -> np.ndarray:
        """
        all readable forcing types in CAMELS-GB

        Returns
        -------
        np.array
            forcing types
        """
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

    def read_gb_gage_flow_forcing(self, gage_id, t_range, var_type):
        """
        Read gage's streamflow or forcing from CAMELS-GB

        Parameters
        ----------
        gage_id
            the station id
        t_range
            the time range, for example, ["1970-10-01", "2015-10-01"]
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
        data_temp = pd.read_csv(gage_file, sep=self.data_file_attr["sep"])
        obs = data_temp[var_type].values
        if var_type in self.target_cols:
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
        read target values; for CAMELS-GB, they are streamflows

        default target_cols is an one-value list
        Notice: the unit of target outputs in different regions are not totally same

        Parameters
        ----------
        gage_id_lst
            station ids
        t_range
            the time range, for example, ["1970-10-01", "2015-10-01"]
        target_cols
            the default is None, but we need at least one default target.
            For CAMELS-GB, it's ["discharge_spec"]
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
            range(len(target_cols)), desc="Read streamflow data of CAMELS-GB"
        ):
            for k in tqdm(range(len(gage_id_lst))):
                data_obs = self.read_gb_gage_flow_forcing(
                    gage_id_lst[k], t_range, target_cols[j]
                )
                y[k, :, j] = data_obs
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
            the time range, for example, ["1970-10-01", "2015-10-01"]
        var_lst
            forcing variable types
        forcing_type
            support for CAMELS-GB, there are only one type: observation
        Returns
        -------
        np.array
            forcing data
        """
        t_range_list = hydro_time.t_range_days(t_range)
        nt = t_range_list.shape[0]
        x = np.full([len(gage_id_lst), nt, len(var_lst)], np.nan)
        for j in tqdm(range(len(var_lst)), desc="Read forcing data of CAMELS-GB"):
            for k in tqdm(range(len(gage_id_lst))):
                data_forcing = self.read_gb_gage_flow_forcing(
                    gage_id_lst[k], t_range, var_lst[j]
                )
                x[k, :, j] = data_forcing
        return x

    def get_attribute_units_dict(self):
        """

        Returns
        -------

        """
        units_dict = {
            "p_mean": "mm/day",
            "pet_mean": "mm/day",
            "aridity": "dimensionless",
            "p_seasonality": "dimensionless",
            "frac_snow": "dimensionless",
            "high_prec_freq": "days/yr",
            "high_prec_dur": "days",
            "high_prec_timing": "season",
            "low_prec_freq": "days/yr",
            "low_prec_dur": "days",
            "low_prec_timing": "season",
            "benchmark_catch": "Y/N",
            "surfacewater_abs": "mm/day",
            "groundwater_abs": "mm/day",
            "discharges": "mm/day",
            "abs_agriculture_perc": "percent",
            "abs_amenities_perc": "percent",
            "abs_energy_perc": "percent",
            "abs_environmental_perc": "percent",
            "abs_industry_perc": "percent",
            "abs_watersupply_perc": "percent",
            "num_reservoir": "dimensionless",
            "reservoir_cap": "ML",
            "reservoir_he": "percent",
            "reservoir_nav": "percent",
            "reservoir_drain": "percent",
            "reservoir_wr": "percent",
            "reservoir_fs": "percent",
            "reservoir_env": "percent",
            "reservoir_nousedata": "percent",
            "reservoir_year_first": "dimensionless",
            "reservoir_year_last": "dimensionless",
            "inter_high_perc": "percent",
            "inter_mod_perc": "percent",
            "inter_low_perc": "percent",
            "frac_high_perc": "percent",
            "frac_mod_perc": "percent",
            "frac_low_perc": "percent",
            "no_gw_perc": "percent",
            "low_nsig_perc": "percent",
            "nsig_low_perc": "percent",
            "q_mean": "mm/day",
            "runoff_ratio": "dimensionless",
            "stream_elas": "dimensionless",
            "slope_fdc": "dimensionless",
            "baseflow_index": "dimensionless",
            "baseflow_index_ceh": "dimensionless",
            "hfd_mean": "days",  # the original unit of this field is "days since 1st October",   ValueError: Failed to decode variable 'hfd_mean': unable to decode time units 'days since 1st October' with 'the default calendar'.
            "Q5": "mm/day",
            "Q95": "mm/day",
            "high_q_freq": "days/yr",
            "high_q_dur": "days",
            "low_q_freq": "days/yr",
            "low_q_dur": "days",
            "zero_q_freq": "percent",
            "station_type": "dimensionless",
            "flow_period_start": "dimensionless",
            "flow_period_end": "dimensionless",
            "flow_perc_complete": "percent",
            "bankfull_flow": "m3 sdimensionless1",
            "structurefull_flow": "m3 sdimensionless1",
            "q5_uncert_upper": "percent",
            "q5_uncert_lower": "percent",
            "q25_uncert_upper": "percent",
            "q25_uncert_lower": "percent",
            "q50_uncert_upper": "percent",
            "q50_uncert_lower": "percent",
            "q75_uncert_upper": "percent",
            "q75_uncert_lower": "percent",
            "q95_uncert_upper": "percent",
            "q95_uncert_lower": "percent",
            "q99_uncert_upper": "percent",
            "q99_uncert_lower": "percent",
            "quncert_meta": "dimensionless",
            "dwood_perc": "percent",
            "ewood_perc": "percent",
            "grass_perc": "percent",
            "shrub_perc": "percent",
            "crop_perc": "percent",
            "urban_perc": "percent",
            "inwater_perc": "percent",
            "bares_perc": "percent",
            "dom_land_cover": "dimensionless",
            "sand_perc": "percent",
            "sand_perc_missing": "percent",
            "silt_perc": "percent",
            "silt_perc_missing": "percent",
            "clay_perc": "percent",
            "clay_perc_missing": "percent",
            "organic_perc": "percent",
            "organic_perc_missing": "percent",
            "bulkdens": "g/cm^3",
            "bulkdens_missing": "percent",
            "bulkdens_5": "g/cm^3",
            "bulkdens_50": "g/cm^3",
            "bulkdens_95": "g/cm^3",
            "tawc": "mm",
            "tawc_missing": "percent",
            "tawc_5": "mm",
            "tawc_50": "mm",
            "tawc_95": "mm",
            "porosity_cosby": "dimensionless",
            "porosity_cosby_missing": "percent",
            "porosity_cosby_5": "dimensionless",
            "porosity_cosby_50": "dimensionless",
            "porosity_cosby_95": "dimensionless",
            "porosity_hypres": "dimensionless",
            "porosity_hypres_missing": "percent",
            "porosity_hypres_5": "dimensionless",
            "porosity_hypres_50": "dimensionless",
            "porosity_hypres_95": "dimensionless",
            "conductivity_cosby": "cm/h",
            "conductivity_cosby_missing": "percent",
            "conductivity_cosby_5": "cm/h",
            "conductivity_cosby_50": "cm/h",
            "conductivity_cosby_95": "cm/h",
            "conductivity_hypres": "cm/h",
            "conductivity_hypres_missing": "percent",
            "conductivity_hypres_5": "cm/h",
            "conductivity_hypres_50": "cm/h",
            "conductivity_hypres_95": "cm/h",
            "root_depth": "m",
            "root_depth_missing": "percent",
            "root_depth_5": "m",
            "root_depth_50": "m",
            "root_depth_95": "m",
            "soil_depth_pelletier": "m",
            "soil_depth_pelletier_missing": "percent",
            "soil_depth_pelletier_5": "m",
            "soil_depth_pelletier_50": "m",
            "soil_depth_pelletier_95": "m",
            "gauge_name": "dimensionless",
            "gauge_lat": "degree",
            "gauge_lon": "degree",
            "gauge_easting": "m",
            "gauge_northing": "m",
            "gauge_elev": "m.a.s.l",
            "area": "km^2",
            "dpsbar": "m/km",
            "elev_mean": "m.a.s.l",
            "elev_min": "m.a.s.l",
            "elev_10": "m.a.s.l",
            "elev_50": "m.a.s.l",
            "elev_90": "m.a.s.l",
            "elev_max": "m.a.s.l",
        }

        return units_dict
