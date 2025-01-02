import logging
import os
import collections
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
from tqdm import tqdm
from hydroutils import hydro_time, hydro_file
from hydrodataset import CACHE_DIR, HydroDataset, CAMELS_REGIONS
from hydrodataset.camels import Camels, time_intersect_dynamic_data
from pandas.api.types import is_string_dtype, is_numeric_dtype

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
        super().__init__(data_path, download, region)

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
        flow_dir_m3s = camels_db.joinpath(
            "02_CAMELS_BR_streamflow_m3s", "02_CAMELS_BR_streamflow_m3s"
        )
        flow_dir_mm_selected_catchments = camels_db.joinpath(
            "03_CAMELS_BR_streamflow_mm_selected_catchments",
            "03_CAMELS_BR_streamflow_mm_selected_catchments",
        )
        flow_dir_simulated = camels_db.joinpath(
            "04_CAMELS_BR_streamflow_simulated",
            "04_CAMELS_BR_streamflow_simulated",
        )

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
                flow_dir_m3s,
                flow_dir_mm_selected_catchments,
                flow_dir_simulated,
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
        return pd.read_csv(camels_file, sep="\s+", dtype={"gauge_id": str})

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

    def read_br_gage_flow(self, gage_id, t_range, flow_type):
        """
        Read gage's streamflow from CAMELS-BR

        Parameters
        ----------
        gage_id
            the station id
        t_range
            the time range, for example, ["1990-01-01", "2000-01-01"]
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
        elif flow_type == "streamflow_simulated":
            flow_type = "simulated_streamflow"
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
            the time range, for example, ["1990-01-01", "2000-01-01"]
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
        y = y * 35.314666721489
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
            the forcing variable type

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
            the time range, for example, ["1990-01-01", "2000-01-01"]
        var_lst
            forcing variable types
        forcing_type
            now only for CAMELS-US, there are three types: daymet, nldas, maurer
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
        gage_dict = self.sites
        camels_str = "camels_br_"
        sep_ = "\s+"
        for key in key_lst:
            data_file = os.path.join(data_folder, camels_str + key + ".txt")
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
