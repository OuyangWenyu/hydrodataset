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

class CamelsCh(Camels):
    def __init__(
        self,
        data_path = os.path.join("camels","camels_ch"),
        download = False,
        region: str = "CH",
    ):
        """
        Initialization for CAMELS-CH dataset

        Parameters
        ----------
        data_path
            where we put the dataset.
            we already set the ROOT directory for hydrodataset,
            so here just set it as a relative path,
            by default "camels/camels_ch"
        download
            if true, download, by default False
        region
            the default is CAMELS-CH
        """
        super().__init__(data_path,download,region)

    def set_data_source_describe(self) -> collections.OrderedDict:
        """
        the files in the dataset and their location in file system

        Returns
        -------
        collections.OrderedDict
            the description for a CAMELS-CH dataset
        """
        camels_db = self.data_source_dir

        if self.region == "CH":
            return self._set_data_source_camelsch_describe(camels_db)
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)

    def _set_data_source_camelsch_describe(self, camels_db):
        # shp file of basins
        camels_shp_file = camels_db.joinpath(
            "catchment_delineations",
            "CAMELS_CH_sub_catchments.shp",
        )
        # flow and forcing data are in a same file
        flow_dir = camels_db.joinpath(
            "timeseries",
            "observation_based",
        )
        forcing_dir = flow_dir
        forcing_types = ["observation", "simulation"]
        # attr
        attr_dir = camels_db.joinpath(
            "static_attributes"
        )
        attr_key_lst = [
            "climate",
            "geology",
            "glacier",
            "humaninfluence",
            "hydrogeology",
            "hydrology",
            "landcover",
            "soil",
            "topographic",
            "catchments", #this file static_attributes\CAMELS_CH_catchments_attributes.csv should be exported manually from catchment_delineations\CAMELS_CH_catchments.shp to replace the static_attributes\CAMELS_CH_sub_catchment_attributes.csv file, for the reason that the sub file do not contain all stations.
        ]
        gauge_id_file = attr_dir.joinpath("CAMELS_CH_hydrology_attributes_obs.csv")

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
        Read the basic information of gages in a CAMELS-CH dataset.

        Returns
        -------
        pd.DataFrame
            basic info of gages
        """
        camels_file = self.data_source_description["CAMELS_GAUGE_FILE"]
        return pd.read_csv(camels_file,sep=",",header=1,dtype={"gauge_id": str})

    def get_constant_cols(self) -> np.ndarray:
        """
        all readable attrs in CAMELS-CH

        Returns
        -------
        np.ndarray
            attribute types
        """
        data_folder = self.data_source_description["CAMELS_ATTR_DIR"]
        return self._get_constant_cols_some(
            data_folder, "CAMELS_CH_","_attributes_obs.csv",","
        )

    def get_relevant_cols(self) -> np.ndarray:
        """
        all readable forcing types in CAMELS-CH

        Returns
        -------
        np.ndarray
            forcing types
        """
        return np.array(
            [
                "waterlevel(m)",
                "precipitation(mm/d)",
                "temperature_min(degC)",
                "temperature_mean(degC)",
                "temperature_max(degC)",
                "rel_sun_dur(%)",
                "swe(mm)",
            ]
        )

    def get_target_cols(self) -> np.ndarray:
        """
        For CAMELS-CH, the target vars are streamflows

        Returns
        -------
        np.ndarray
            streamflow types
        """
        return np.array(["discharge_vol(m3/s)", "discharge_spec(mm/d)"])

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

    def read_ch_gage_flow_forcing(self, gage_id, t_range, var_type):
        """
        Read gage's streamflow or forcing from CAMELS-CH

        Parameters
        ----------
        gage_id
            the station id
        t_range
            the time range, for example, ["1981-01-01","2020-12-31"]
        var_type
            flow type: "discharge_vol(m3/s)", "discharge_spec(mm/d)"
            forcing type: "waterlevel(m)", "precipitation(mm/d)", "temperature_min(degC)", "temperature_mean(degC)", "temperature_max(degC)", "rel_sun_dur(%)", "swe(mm)"

        Returns
        -------
        np.array
            streamflow or forcing data of one station for a given time range
        """
        logging.debug("reading %s streamflow data", gage_id)
        gage_file = os.path.join(
            self.data_source_description["CAMELS_FLOW_DIR"],
            "CAMELS_CH_obs_based_" + gage_id + ".csv",
        )
        data_temp = pd.read_csv(gage_file, sep=",")
        obs = data_temp[var_type].values
        if var_type in ["discharge_vol(m3/s)", "discharge_spec(mm/d)"]:
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
        read target values. for CAMELS-CH, they are streamflows.

        default target_cols is an one-value list
        Notice, the unit of target outputs in different regions are not totally same

        Parameters
        ----------
        gage_id_lst
            station ids
        t_range
            the time range, for example, ["1981-01-01","2020-12-31"]
        target_cols
            the default is None, but we need at least one default target.
            For CAMELS-CH, it's ["discharge_vol(m3/s)"]
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
            range(len(target_cols)), desc="Read streamflow data of CAMELS-CH"
        ):
            for k in tqdm(range(len(gage_id_lst))):
                data_obs = self.read_ch_gage_flow_forcing(
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
            the time range, for example, ["1981-01-01","2020-12-31"]
        var_lst
            forcing variable type: "waterlevel(m)", "precipitation(mm/d)", "temperature_min(degC)", "temperature_mean(degC)", "temperature_max(degC)", "rel_sun_dur(%)", "swe(mm)"
        forcing_type
            support for CAMELS-CH, there are two types: observation, simulation
        Returns
        -------
        np.array
            forcing data
        """
        t_range_list = hydro_time.t_range_days(t_range)
        nt = t_range_list.shape[0]
        x = np.full([len(gage_id_lst), nt, len(var_lst)], np.nan)
        for j in tqdm(range(len(var_lst)), desc="Read forcing data of CAMELS-CH"):
            for k in tqdm(range(len(gage_id_lst))):
                data_forcing = self.read_ch_gage_flow_forcing(
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
        camels_str = "CAMELS_CH_"
        sep_ = ","
        for key in key_lst:
            if key in ["climate", "hydrology"]:
                data_file = os.path.join(data_folder, camels_str + key + "_attributes_obs.csv")
            else:
                data_file = os.path.join(data_folder, camels_str + key + "_attributes.csv")
            header_ = 1
            if key in ["catchments",]:
                header_ = 0
            data_temp = pd.read_csv(data_file, sep=sep_,header=header_)  # if a UnicodeDecodeError bug appeared, a manually encoding transform for camels_ch\static_attributes\CAMELS_CH_topographic_attributes.csv is need, encoding->convert to UTF-8 in Notepad++.
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
        return out, var_lst, var_dict, f_dict  #

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
            if true, return var_dict and f_dict for CAMELS-CH
        Returns
        -------
        Union[tuple, np.array]
            if attr var type is str, return factorized data.
            When we need to know what a factorized value represents, we need return a tuple;
            otherwise just return an array
        """
        attr_all, var_lst_all, var_dict, f_dict = self.read_attr_all() #
        ind_var = [var_lst_all.index(var) for var in var_lst]
        id_lst_all = self.read_object_ids()
        # Notice the sequence of station ids ! Some id_lst_all are not sorted, so don't use np.intersect1d
        ind_grid = [id_lst_all.tolist().index(tmp) for tmp in gage_id_lst]
        temp = attr_all[ind_grid, :]
        out = temp[:, ind_var]
        return (out, var_dict, f_dict) if is_return_dict else out

    def read_area(self, gage_id_lst) -> np.ndarray:
        return self.read_constant_cols(gage_id_lst, ["area"], is_return_dict=False)

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
        cache_npy_file = CACHE_DIR.joinpath("camels_ch_forcing.npy")  # "C:\Users\xxxx\AppData\Local\hydro\Cache\camels_ch_forcing.json"
        json_file = CACHE_DIR.joinpath("camels_ch_forcing.json")
        variables = self.get_relevant_cols()
        basins = self.sites["gauge_id"].values
        t_range = ["1981-01-01","2020-12-31"]
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
        cache_npy_file = CACHE_DIR.joinpath("camels_ch_streamflow.npy")
        json_file = CACHE_DIR.joinpath("camels_ch_streamflow.json")
        variables = self.get_target_cols()
        basins = self.sites["gauge_id"].values
        t_range = ["1981-01-01","2020-12-31"]
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
        attrs_df = pd.DataFrame(data=attr_all[0:,0:],columns=var_lst_all)

        # unify id to basin
        attrs_df.index.name = "basin"
        # We use xarray dataset to cache all data
        ds_from_df = attrs_df.to_xarray()  # *
        units_dict = {
            "ind_start_date": "dimensionless",
            "ind_end_date": "dimensionless",
            "ind_number_of_years": "dimensionless",
            "p_mean": "mm/d",
            "pet_mean": "mm/d",
            "aridity": "dimensionless",
            "p_seasonality": "dimensionless",
            "frac_snow": "dimensionless",
            "high_prec_freq": "d/yr",
            "high_prec_dur": "d",
            "high_prec_timing": "season",
            "low_prec_freq": "d/yr",
            "low_prec_dur": "d",
            "low_prec_timing": "season",
            "geo_porosity": "dimensionless",
            "geo_log10_permeability": "log10(m^2)",
            "unconsol_sediments": "%",
            "siliciclastic_sedimentary": "%",
            "mixed_sedimentary": "%",
            "carbonate_sedimentary": "%",
            "pyroclastic": "%",
            "acid_volcanic": "%",
            "basic_volcanic": "%",
            "acid_plutonic": "%",
            "intermediate_plutonic": "%",
            "basic_plutonic": "%",
            "metamorphics": "%",
            "water_geo": "%",
            "ice_geo": "%",
            "glac_area": "km^2",
            "glac_vol": "km^3",
            "glac_mass": "MT (10^6 metric tons)",
            "glac_area_neighbours": "km^2",
            "n_inhabitants": "dimensionless",
            "dens_inhabitants": "km^2",
            "hp_count": "dimensionless",
            "hp_qturb": "m^3/s",
            "hp_inst_turb": "MW",
            "hp_max_power": "MW",
            "num_reservoir": "dimensionless",
            "reservoir_cap": "ML",
            "reservoir_he": "%",
            "reservoir_fs": "%",
            "reservoir_irr": "%",
            "reservoir_nousedata": "%",
            "reservoir_year_first": "%",
            "reservoir_year_last": "%",
            "unconsol_coarse_perc": "%",
            "unconsol_medium_perc": "%",
            "unconsol_fine_perc": "%",
            "unconsol_imperm_perc": "%",
            "hardrock_perc": "%",
            "hardrock_imperm_perc": "%",
            "karst_perc": "%",
            "water_perc": "%",
            "null_perc": "%",
            "ext_area_perc": "%",
            "sign_start_date": "dimensionless",
            "sign_end_date": "dimensionless",
            "sign_number_of_years": "dimensionless",
            "q_mean": "mm/d",
            "runoff_ratio": "dimensionless",
            "stream_elas": "dimensionless",
            "slope_fdc": "dimensionless",
            "baseflow_index_landson": "dimensionless",
            "hfd_mean": "d",
            "Q5": "mm/d",
            "Q95": "mm/d",
            "high_q_freq": "d/yr",
            "high_q_dur": "d",
            "low_q_freq": "d/yr",
            "low_q_dur": "d",
            "zero_q_freq": "dimensionless",
            "crop_perc": "%",
            "grass_perc": "%",
            "scrub_perc": "%",
            "dwood_perc": "%",
            "mixed_wood_perc": "%",
            "ewood_perc": "%",
            "wetlands_perc": "%",
            "inwater_perc": "%",
            "ice_perc": "%",
            "loose_rock_perc": "%",
            "rock_perc": "%",
            "urban_perc": "%",
            "dom_land_cover": "dimensionless",
            "sand_perc": "%",
            "sand_perc_5": "%",
            "sand_perc_25": "%",
            "sand_perc_50": "%",
            "sand_perc_75": "%",
            "sand_perc_90": "%",
            "sand_perc_skewness": "%",
            "sand_perc_missing": "%",
            "silt_perc": "%",
            "silt_perc_5": "%",
            "silt_perc_25": "%",
            "silt_perc_50": "%",
            "silt_perc_75": "%",
            "silt_perc_90": "%",
            "silt_perc_skewness": "%",
            "silt_perc_missing": "%",
            "clay_perc": "%",
            "clay_perc_5": "%",
            "clay_perc_25": "%",
            "clay_perc_50": "%",
            "clay_perc_75": "%",
            "clay_perc_90": "%",
            "clay_perc_skewness": "%",
            "clay_perc_missing": "%",
            "organic_perc": "%",
            "organic_perc_5": "%",
            "organic_perc_25": "%",
            "organic_perc_50": "%",
            "organic_perc_75": "%",
            "organic_perc_90": "%",
            "organic_perc_skewness": "%",
            "organic_perc_missing": "%",
            "bulk_dens": "g/cm^3",
            "bulk_dens_5": "g/cm^3",
            "bulk_dens_25": "g/cm^3",
            "bulk_dens_50": "g/cm^3",
            "bulk_dens_75": "g/cm^3",
            "bulk_dens_90": "g/cm^3",
            "bulk_dens_skewness": "g/cm^3",
            "bulk_dens_missing": "g/cm^3",
            "tot_avail_water": "mm",
            "tot_avail_water_5": "mm",
            "tot_avail_water_25": "mm",
            "tot_avail_water_50": "mm",
            "tot_avail_water_75": "mm",
            "tot_avail_water_90": "mm",
            "tot_avail_water_skewness": "mm",
            "tot_avail_water_missing": "mm",
            "root_depth": "m",
            "root_depth_5": "m",
            "root_depth_25": "m",
            "root_depth_50": "m",
            "root_depth_75": "m",
            "root_depth_90": "m",
            "root_depth_skewness": "m",
            "root_depth_missing": "m",
            "coarse_fragm_perc": "%",
            "coarse_fragm_perc_5": "%",
            "coarse_fragm_perc_25": "%",
            "coarse_fragm_perc_50": "%",
            "coarse_fragm_perc_75": "%",
            "coarse_fragm_perc_90": "%",
            "coarse_fragm_perc_skewness": "%",
            "coarse_fragm_perc_missing": "%",
            "porosity": "dimensionless",
            "porosity_5": "dimensionless",
            "porosity_25": "dimensionless",
            "porosity_50": "dimensionless",
            "porosity_75": "dimensionless",
            "porosity_90": "dimensionless",
            "porosity_skewness": "dimensionless",
            "porosity_missing": "dimensionless",
            "conductivity": "cm/h",
            "conductivity_5": "cm/h",
            "conductivity_25": "cm/h",
            "conductivity_50": "cm/h",
            "conductivity_75": "cm/h",
            "conductivity_90": "cm/h",
            "conductivity_skewness": "cm/h",
            "conductivity_missing": "cm/h",
            "country": "dimensionless",
            "gauge_name": "dimensionless",
            "water_body_name": "dimensionless",
            "id6": "dimensionless",
            "water_body_type": "dimensionless",
            "gauge_londegree": "degree",
            "gauge_latdegree": "degree",
            "gauge_easting": "m",
            "gauge_northing": "m",
            "gauge_elevation": "m.a.s.l.",
            "area": "km^2",
            "elev_mean": "m.a.s.l.",
            "elev_min": "m.a.s.l.",
            "elev_percentile10": "m.a.s.l.",
            "elev_percentile25": "m.a.s.l.",
            "elev_percentile50": "m.a.s.l.",
            "elev_percentile75": "m.a.s.l.",
            "elev_percentile90": "m.a.s.l.",
            "elev_max": "m.a.s.l.",
            "slope_mean": "degree",
            "flat_area_perc": "%",
            "steep_area_perc": "%",
            "ID6": "dimensionless",
            "gauge_name": "dimensionless",
            "water_body": "dimensionless",
            "type": "dimensionless",
            "country": "dimensionless",
            "Shape_Leng": "m",
            "Shape_Area": "m^2",
            "ORIG_FID": "dimensionless",
        }

        # Assign units to the variables in the Dataset
        for var_name in units_dict:
            if var_name in ds_from_df.data_vars:
                ds_from_df[var_name].attrs["units"] = units_dict[var_name]

        return ds_from_df

    def cache_forcing_xrdataset(self):
        """Save all basin-forcing data in a netcdf file in the cache directory.

        """
        cache_npy_file = CACHE_DIR.joinpath("camels_ch_forcing.npy")
        json_file = CACHE_DIR.joinpath("camels_ch_forcing.json")
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

        units = ["m", "mm/day", "°C", "°C", "°C", "%", "mm"]
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
        cache_npy_file = CACHE_DIR.joinpath("camels_ch_streamflow.npy")
        json_file = CACHE_DIR.joinpath("camels_ch_streamflow.json")
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
        ds_attr.to_netcdf(CACHE_DIR.joinpath("camelsch_attributes.nc"))
        ds_streamflow = self.cache_streamflow_xrdataset()
        ds_forcing = self.cache_forcing_xrdataset()
        ds = xr.merge([ds_streamflow, ds_forcing])
        ds.to_netcdf(CACHE_DIR.joinpath("camelsch_timeseries.nc"))  # todo: Forward slashes '/' are not allowed in variable and dimension names (got 'precipitation(mm/d)'). Forward slashes are used as hierarchy-separators for HDF5-based files ('netcdf4'/'h5netcdf').
