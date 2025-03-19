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


class CamelsBr(Camels):
    def __init__(
        self,
        data_path=os.path.join("camels", "camels_br"),
        download=False,
        region: str = "BR",
        gauge_id_tag: str ="gauge_id",
    ):
        """
        Initialization for CAMELS-BR dataset

        Parameters
        ----------
        data_path
            where we put the dataset.
            we already set the ROOT directory for hydrodataset,
            so here just set it as a relative path,
            by default "camels/camels_br"
        download
            if true, download, by default False
        region
            the default is CAMELS-BR
        """
        super().__init__(data_path, download, region, gauge_id_tag)

    def set_data_source_describe(self) -> collections.OrderedDict:
        """
        the files in the dataset and their location in file system

        Returns
        -------
        collections.OrderedDict
            the description for a CAMELS-BR dataset
        """
        camels_db = self.data_source_dir

        return self._set_data_source_camelsbr_describe(camels_db)

    def _set_data_source_camelsbr_describe(self, camels_db):
        # attr
        attr_dir = camels_db.joinpath(
            "01_CAMELS_BR_attributes", "01_CAMELS_BR_attributes"
        )
        # we don't need the location attr file
        attr_key_lst = [
            "climate",
            "geology",
            "human_intervention",
            "hydrology",
            "land_cover",
            "quality_check",
            "soil",
            "topography",
        ]
        # id and name, there are two types stations in CAMELS_BR, and we only chose the 897-stations version
        gauge_id_file = attr_dir.joinpath("camels_br_topography.txt")
        # shp file of basins
        camels_shp_file = camels_db.joinpath(
            "14_CAMELS_BR_catchment_boundaries",
            "14_CAMELS_BR_catchment_boundaries",
            "camels_br_catchments.shp",
        )
        # config of flow data
        # flow_dir_m3s = camels_db.joinpath(
        #     "02_CAMELS_BR_streamflow_m3s", "02_CAMELS_BR_streamflow_m3s"
        # )
        flow_dir_mm_selected_catchments = camels_db.joinpath(
            "03_CAMELS_BR_streamflow_mm_selected_catchments",
            "03_CAMELS_BR_streamflow_mm_selected_catchments",
        )
        # flow_dir_simulated = camels_db.joinpath(
        #     "04_CAMELS_BR_streamflow_simulated",
        #     "04_CAMELS_BR_streamflow_simulated",
        # )

        # forcing
        forcing_dir_precipitation_chirps = camels_db.joinpath(
            "05_CAMELS_BR_precipitation_chirps",
            "05_CAMELS_BR_precipitation_chirps",
        )
        forcing_dir_precipitation_mswep = camels_db.joinpath(
            "06_CAMELS_BR_precipitation_mswep",
            "06_CAMELS_BR_precipitation_mswep",
        )
        forcing_dir_precipitation_cpc = camels_db.joinpath(
            "07_CAMELS_BR_precipitation_cpc",
            "07_CAMELS_BR_precipitation_cpc",
        )
        forcing_dir_evapotransp_gleam = camels_db.joinpath(
            "08_CAMELS_BR_evapotransp_gleam",
            "08_CAMELS_BR_evapotransp_gleam",
        )
        forcing_dir_evapotransp_mgb = camels_db.joinpath(
            "09_CAMELS_BR_evapotransp_mgb",
            "09_CAMELS_BR_evapotransp_mgb",
        )
        forcing_dir_potential_evapotransp_gleam = camels_db.joinpath(
            "10_CAMELS_BR_potential_evapotransp_gleam",
            "10_CAMELS_BR_potential_evapotransp_gleam",
        )
        forcing_dir_temperature_min_cpc = camels_db.joinpath(
            "11_CAMELS_BR_temperature_min_cpc",
            "11_CAMELS_BR_temperature_min_cpc",
        )
        forcing_dir_temperature_mean_cpc = camels_db.joinpath(
            "12_CAMELS_BR_temperature_mean_cpc",
            "12_CAMELS_BR_temperature_mean_cpc",
        )
        forcing_dir_temperature_max_cpc = camels_db.joinpath(
            "13_CAMELS_BR_temperature_max_cpc",
            "13_CAMELS_BR_temperature_max_cpc",
        )
        return collections.OrderedDict(
            CAMELS_DIR=camels_db,
            CAMELS_FLOW_DIR=[
                # flow_dir_m3s,
                flow_dir_mm_selected_catchments,
                # flow_dir_simulated,
            ],
            CAMELS_FORCING_DIR=[
                forcing_dir_precipitation_chirps,
                forcing_dir_precipitation_mswep,
                forcing_dir_precipitation_cpc,
                forcing_dir_evapotransp_gleam,
                forcing_dir_evapotransp_mgb,
                forcing_dir_potential_evapotransp_gleam,
                forcing_dir_temperature_min_cpc,
                forcing_dir_temperature_mean_cpc,
                forcing_dir_temperature_max_cpc,
            ],
            CAMELS_ATTR_DIR=attr_dir,
            CAMELS_ATTR_KEY_LST=attr_key_lst,
            CAMELS_GAUGE_FILE=gauge_id_file,
            CAMELS_BASINS_SHP_FILE=camels_shp_file,
        )

    def read_site_info(self) -> pd.DataFrame:
        """
        Read the basic information of gages in a CAMELS-BR dataset

        Returns
        -------
        pd.DataFrame
            basic info of gages
        """
        camels_file = self.data_source_description["CAMELS_GAUGE_FILE"]
        return pd.read_csv(camels_file, sep="\s+", dtype={self.gauge_id_tag: str})

    def get_constant_cols(self) -> np.ndarray:
        """
        all readable attrs in CAMELS-BR

        Returns
        -------
        np.array
            attribute types
        """
        data_folder = self.data_source_description["CAMELS_ATTR_DIR"]
        return self._get_constant_cols_some(data_folder, "camels_br_", ".txt", "\s+")

    def get_relevant_cols(self) -> np.ndarray:
        """
        all readable forcing types in CAMELS-BR

        Returns
        -------
        np.array
            forcing types
        """
        return np.array(
            [
                str(forcing_dir).split(os.sep)[-1][13:]
                for forcing_dir in self.data_source_description["CAMELS_FORCING_DIR"]
            ]
        )

    def get_target_cols(self) -> np.ndarray:
        """
        For CAMELS-BR, the target vars are streamflows

        Returns
        -------
        np.array
            streamflow types
        """
        return np.array(
            [
                str(flow_dir).split(os.sep)[-1][13:]
                for flow_dir in self.data_source_description["CAMELS_FLOW_DIR"]
            ]
        )

    def read_br_gage_flow(self, gage_id, t_range, flow_type):
        """
        Read gage's streamflow from CAMELS-BR

        Parameters
        ----------
        gage_id
            the station id
        t_range
            the time range, for example, ["1995-01-01", "2015-01-01"]
        flow_type
            "streamflow_m3s" or "streamflow_mm_selected_catchments" or "streamflow_simulated"

        Returns
        -------
        np.array
            streamflow data of one station for a given time range
        """
        logging.debug("reading %s streamflow data", gage_id)
        dir_ = [
            str(flow_dir)
            for flow_dir in self.data_source_description["CAMELS_FLOW_DIR"]
            if flow_type in str(flow_dir)
        ][0]
        if flow_type == "streamflow_mm_selected_catchments":
            flow_type = "streamflow_mm"
        # elif flow_type == "streamflow_simulated":
        #     flow_type = "simulated_streamflow"
        gage_file = os.path.join(dir_, gage_id + "_" + flow_type + ".txt")
        data_temp = pd.read_csv(gage_file, sep=r"\s+")
        obs = data_temp.iloc[:, 3].values
        obs[obs < 0] = np.nan
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
        read target values; for CAMELS-BR, they are streamflows

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
            For CAMELS-BR, it's ["streamflow_mmd"]
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
            range(len(target_cols)), desc="Read streamflow data of CAMELS-BR"
        ):
            for k in tqdm(range(len(gage_id_lst))):
                data_obs = self.read_br_gage_flow(
                    gage_id_lst[k], t_range, target_cols[j]
                )
                y[k, :, j] = data_obs
        # Keep unit of streamflow unified: we use ft3/s here
        # other units are m3/s -> ft3/s
        y = self.unit_convert_streamflow_m3tofoot3(y)
        return y

    def read_br_basin_forcing(self, gage_id, t_range, var_type) -> np.array:
        """
        Read one forcing data for a basin in CAMELS_BR

        Parameters
        ----------
        gage_id
            basin id
        t_range
            the time range, for example, ["1995-01-01", "2005-01-01"]
        var_type
            the forcing variable type, "precipitation_chirps", "precipitation_mswep", "precipitation_cpc", "evapotransp_gleam", "evapotransp_mgb",
                   "potential_evapotransp_gleam", "temperature_min_cpc", "temperature_mean_cpc", "temperature_max_cpc"

        Returns
        -------
        np.array
            one type forcing data of a basin in a given time range
        """
        dir_ = [
            str(_dir)
            for _dir in self.data_source_description["CAMELS_FORCING_DIR"]
            if var_type in str(_dir)
        ][0]
        if var_type in [
            "temperature_min_cpc",
            "temperature_mean_cpc",
            "temperature_max_cpc",
        ]:
            var_type = var_type[:-4]
        gage_file = os.path.join(dir_, gage_id + "_" + var_type + ".txt")
        data_temp = pd.read_csv(gage_file, sep=r"\s+")
        obs = data_temp.iloc[:, 3].values
        df_date = data_temp[["year", "month", "day"]]
        date = pd.to_datetime(df_date).values.astype("datetime64[D]")
        return time_intersect_dynamic_data(obs, date, t_range)

    def read_relevant_cols(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        forcing_type="daymet",
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
            forcing variable types
        forcing_type
            now only for CAMELS-BR, there are only one type: observation
        Returns
        -------
        np.array
            forcing data
        """
        t_range_list = hydro_time.t_range_days(t_range)
        nt = t_range_list.shape[0]
        x = np.full([len(gage_id_lst), nt, len(var_lst)], np.nan)
        for j in tqdm(range(len(var_lst)), desc="Read forcing data of CAMELS-BR"):
            for k in tqdm(range(len(gage_id_lst))):
                data_obs = self.read_br_basin_forcing(
                    gage_id_lst[k], t_range, var_lst[j]
                )
                x[k, :, j] = data_obs
        return x

    def read_attr_all(self):
        data_folder = self.data_source_description["CAMELS_ATTR_DIR"]
        key_lst = self.data_source_description["CAMELS_ATTR_KEY_LST"]
        f_dict = {}
        var_dict = {}
        var_lst = []
        out_lst = []
        camels_str = "camels_br_"
        sep_ = "\s+"
        for key in key_lst:
            data_file = os.path.join(data_folder, camels_str + key + ".txt")
            data_temp = pd.read_csv(data_file, sep=sep_)
            var_lst_temp = list(data_temp.columns[1:])
            var_dict[key] = var_lst_temp
            var_lst.extend(var_lst_temp)
            k = 0
            out_temp = np.full([self.n_gage, len(var_lst_temp)], np.nan)
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
            if true, return var_dict and f_dict for CAMELS_US
        Returns
        -------
        Union[tuple, np.array]
            if attr var type is str, return factorized data.
            When we need to know what a factorized value represents, we need return a tuple;
            otherwise just return an array
        """
        attr_all, var_lst_all, var_dict, f_dict = self.read_attr_all()
        ind_var = [var_lst_all.index(var) for var in var_lst]
        id_lst_all = self.gage
        # Notice the sequence of station ids ! Some id_lst_all are not sorted, so don't use np.intersect1d
        ind_grid = [id_lst_all.tolist().index(tmp) for tmp in gage_id_lst]
        temp = attr_all[ind_grid, :]
        out = temp[:, ind_var]
        return (out, var_dict, f_dict) if is_return_dict else out

    def read_area(self, gage_id_lst) -> np.ndarray:
        return self.read_attr_xrdataset(gage_id_lst, ["area"], is_return_dict=False)

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
        data = self.read_attr_xrdataset(
            gage_id_lst,
            ["p_mean"],
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
        cache_npy_file = CACHE_DIR.joinpath("camels_br_forcing.npy")
        json_file = CACHE_DIR.joinpath("camels_br_forcing.json")
        variables = self.get_relevant_cols()
        basins = self.gage
        t_range = ["1995-01-01", "2015-01-01"]
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
        cache_npy_file = CACHE_DIR.joinpath("camels_br_streamflow.npy")
        json_file = CACHE_DIR.joinpath("camels_br_streamflow.json")
        variables = self.get_target_cols()
        basins = self.gage
        t_range = ["1995-01-01", "2015-01-01"]
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
        basins = self.gage
        attrs_df = pd.DataFrame(data=attr_all[0:, 0:], index=basins, columns=var_lst_all)

        # unify id to basin
        attrs_df.index.name = "basin"
        # We use xarray dataset to cache all data
        ds_from_df = attrs_df.to_xarray()
        units_dict = {
            "p_mean": "mm/day",
            "pet_mean": "mm/day",
            "et_mean": "mm/day",
            "aridity": "dimensionless",
            "p_seasonality": "dimensionless",
            "asynchronicity": "dimensionless",
            "frac_snow": "dimensionless",
            "high_prec_freq": "days/yr",
            "high_prec_dur": "days",
            "high_prec_timing": "season",
            "low_prec_freq": "days/yr",
            "low_prec_dur": "days",
            "low_prec_timing": "season",
            "geol_class_1st": "dimensionless",
            "geol_class_1st_perc": "percent",
            "geol_class_2nd": "dimensionless",
            "geol_class_2nd_perc": "percent",
            "carb_rocks_perc": "percent",
            "geol_porosity": "dimensionless",
            "geol_permeability": "m^2",
            "consumptive_use": "mm/yr",
            "consumptive_use_perc": "percent",
            "reservoirs_vol": "10^6 m^3",
            "regulation_degree": "percent",
            "q_mean": "mm/day",
            "runoff_ratio": "dimensionless",
            "stream_elas": "dimensionless",
            "slope_fdc": "dimensionless",
            "baseflow_index": "dimensionless",
            "hfd_mean": "day of the year",
            "Q5": "mm/day",
            "Q95": "mm/day",
            "high_q_freq": "days/yr",
            "high_q_dur": "days",
            "low_q_freq": "days/yr",
            "low_q_dur": "days",
            "zero_q_freq": "days/yr",
            "crop_perc": "percent",
            "crop_mosaic_perc": "percent",
            "forest_perc": "percent",
            "shrub_perc": "percent",
            "grass_perc": "percent",
            "barren_perc": "percent",
            "imperv_perc": "percent",
            "wet_perc": "percent",
            "snow_perc": "percent",
            "dom_land_cover": "dimensionless",
            "dom_land_cover_perc": "percent",
            "gauge_name": "dimensionless",
            "gauge_region": "dimensionless",
            "gauge_lat": "degree North",
            "gauge_lon": "degree East",
            "area_ana": "km^2",
            "area_gsim": "km^2",
            "area_gsim_quality": "km^2",
            "q_quality_control_perc": "percent",
            "q_stream_stage_perc": "percent",
            "sand_perc": "percent",
            "silt_perc": "percent",
            "clay_perc": "percent",
            "org_carbon_content": "g/kg",
            "bedrock_depth": "cm",
            "water_table_depth": "cm",
            "elev_gauge": "m.a.s.l.",
            "elev_mean": "m.a.s.l.",
            "slope_mean": "m/km",
            "area": "km^2",
        }

        # Assign units to the variables in the Dataset
        for var_name in units_dict:
            if var_name in ds_from_df.data_vars:
                ds_from_df[var_name].attrs["units"] = units_dict[var_name]

        return ds_from_df

    def cache_forcing_xrdataset(self):
        """Save all basin-forcing data in a netcdf file in the cache directory.

        """
        cache_npy_file = CACHE_DIR.joinpath("camels_br_forcing.npy")
        json_file = CACHE_DIR.joinpath("camels_br_forcing.json")
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

        units = ["mm/d", "mm/d", "mm/d", "mm/d", "mm/d", "mm/d", "°C", "°C", "°C"]
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
        cache_npy_file = CACHE_DIR.joinpath("camels_br_streamflow.npy")
        json_file = CACHE_DIR.joinpath("camels_br_streamflow.json")
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
                # "ET": (
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
