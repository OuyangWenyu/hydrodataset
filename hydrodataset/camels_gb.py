import logging
import os
import collections
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
from tqdm import tqdm
import xarray as xr
from hydroutils import hydro_time, hydro_file
from hydrodataset import CACHE_DIR, HydroDataset, CAMELS_REGIONS
from hydrodataset.camels import Camels, time_intersect_dynamic_data
from pandas.api.types import is_string_dtype, is_numeric_dtype
import json
import warnings

CAMELS_NO_DATASET_ERROR_LOG = (
    "We cannot read this dataset now. Please check if you choose correctly:\n"
    + str(CAMELS_REGIONS)
)


class CamelsGb(Camels):
    def __init__(
        self,
        data_path=os.path.join("camels", "camels_gb"),
        download=False,
        region: str = "GB",
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
        super().__init__(data_path, download, region)

    def set_data_source_describe(self) -> collections.OrderedDict:
        """
        the files in the dataset and their location in file system

        Returns
        -------
        collections.OrderedDict
            the description for a CAMELS-GB dataset
        """
        camels_db = self.data_source_dir
        return self._set_data_source_camelsgb_describe(camels_db)

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

    def read_site_info(self) -> pd.DataFrame:
        """
        Read the basic information of gages in a CAMELS-GB dataset

        Returns
        -------
        pd.DataFrame
            basic info of gages
        """
        camels_file = self.data_source_description["CAMELS_GAUGE_FILE"]
        return pd.read_csv(camels_file, sep=",", dtype={"gauge_id": str})

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

    def get_target_cols(self) -> np.ndarray:
        """
        For CAMELS-GB, the target vars are streamflows

        Returns
        -------
        np.array
            streamflow types
        """
        return np.array(["discharge_spec", "discharge_vol"])

    def read_gb_gage_flow_forcing(self, gage_id, t_range, var_type):
        """
        Read gage's streamflow or forcing from CAMELS-GB

        Parameters
        ----------
        gage_id
            the station id
        t_range
            the time range, for example, ["1970-10-01", "2015-09-30"]
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
        read target values; for CAMELS-GB, they are streamflows

        default target_cols is an one-value list
        Notice: the unit of target outputs in different regions are not totally same

        Parameters
        ----------
        gage_id_lst
            station ids
        t_range
            the time range, for example, ["1970-10-01", "2015-09-30"]
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
            the time range, for example, ["1970-10-01", "2015-09-30"]
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

    def read_attr_all(self):
        data_folder = self.data_source_description["CAMELS_ATTR_DIR"]
        key_lst = self.data_source_description["CAMELS_ATTR_KEY_LST"]
        f_dict = {}
        var_dict = {}
        var_lst = []
        out_lst = []
        gage_dict = self.sites
        camels_str = "CAMELS_GB_"
        sep_ = ","
        for key in key_lst:
            data_file = os.path.join(data_folder, camels_str + key + "_attributes.csv")
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

    def read_area(self, gage_id_lst) -> np.ndarray:
        return self.read_constant_cols(gage_id_lst, ["area"], is_return_dict=False)  # todo: area

    def read_mean_prcp(self, gage_id_lst, unit="mm/d") -> xr.Dataset:
        """Read mean precipitation data

        Parameters
        ----------
        gage_id_lst : list
            station ids
        unit : str, optional
            the unit of mean_prcp, by default "mm/d"

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
        cache_npy_file = CACHE_DIR.joinpath("camels_gb_forcing.npy")
        json_file = CACHE_DIR.joinpath("camels_gb_forcing.json")
        variables = self.get_relevant_cols()
        basins = self.sites["gauge_id"].values
        t_range = ["1970-10-01", "2015-09-30"]
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
        cache_npy_file = CACHE_DIR.joinpath("camels_gb_streamflow.npy")
        json_file = CACHE_DIR.joinpath("camels_gb_streamflow.json")
        variables = self.get_target_cols()
        basins = self.sites["gauge_id"].values
        t_range = ["1970-10-01", "2015-09-30"]
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
            "hfd_mean": "days since 1st October",
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

        # Assign units to the variables in the Dataset
        for var_name in units_dict:
            if var_name in ds_from_df.data_vars:
                ds_from_df[var_name].attrs["units"] = units_dict[var_name]

        return ds_from_df

    def cache_forcing_xrdataset(self):
        """Save all basin-forcing data in a netcdf file in the cache directory.

        """
        cache_npy_file = CACHE_DIR.joinpath("camels_gb_forcing.npy")
        json_file = CACHE_DIR.joinpath("camels_gb_forcing.json")
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

        units = ["mm/day", "mm/day", "°C", "mm/day", "m^3/s", "mm/day", "g/kg", "W/m^2", "W/m^2", "m/s"]
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
        cache_npy_file = CACHE_DIR.joinpath("camels_gb_streamflow.npy")
        json_file = CACHE_DIR.joinpath("camels_gb_streamflow.json")
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
        ds_attr.to_netcdf(CACHE_DIR.joinpath("camelsgb_attributes.nc"))
        ds_streamflow = self.cache_streamflow_xrdataset()
        ds_forcing = self.cache_forcing_xrdataset()
        ds = xr.merge([ds_streamflow, ds_forcing])
        ds.to_netcdf(CACHE_DIR.joinpath("camelsgb_timeseries.nc"))
