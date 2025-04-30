import os
import logging
import collections
import pandas as pd
import numpy as np
import xarray as xr
from typing import Union
from tqdm import tqdm
from hydroutils import hydro_time
from hydrodataset import CACHE_DIR, CAMELS_REGIONS
from hydrodataset.camels import Camels, time_intersect_dynamic_data
from pandas.api.types import is_string_dtype, is_numeric_dtype
import json

CAMELS_NO_DATASET_ERROR_LOG = (
    "We cannot read this dataset now. Please check if you choose correctly:\n"
    + str(CAMELS_REGIONS)
)

camelsystl_arg = {
    "forcing_type": "observation",
    "gauge_id_tag": "gauge_id",
    "area_tag": ["area", ],
    "meanprcp_unit_tag": [["p_mean"], "mm/d"],
    "time_range": {
        "observation": ["1990-01-01","1994-01-01"],
    },
    "target_cols": ["discharge_vol",],
    "b_nestedness": False,
    "forcing_unit": ["mm", "mm", "m3/s"],
    "data_file_attr": {
        "sep": ",",
        "header": 0,
        "attr_file_str": ["camels_ystl_", ".csv"]
    },
}

class CamelsYstl(Camels):
    def __init__(
        self,
        data_path = os.path.join("camels","camels_ystl"),
        download = False,
        region: str = "YSTL",
        arg: dict = camelsystl_arg,
    ):
        """
        Initialization for CAMELS-YSTL dataset

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
        super().__init__(data_path, download, region, arg)

    def _set_data_source_camels_describe(self, camels_db):
        # shp file of basins
        camels_shp_file = camels_db
        # flow and forcing data are in a same file
        flow_dir = camels_db
        forcing_dir = flow_dir
        # attr
        attr_dir = camels_db
        attr_key_lst = [
            "climate",
        ]
        gauge_id_file = attr_dir.joinpath("camels_ystl_climate.csv")
        nestedness_information_file = None
        base_url = "https://zenodo.org/records/15025258"
        download_url_lst = [
            f"{base_url}/files/camels_ch.zip",
            f"{base_url}/files/camels_ch_data_description.pdf",
            f"{base_url}/files/Caravan_extension_CH.zip",
        ]

        return collections.OrderedDict(
            CAMELS_DIR = camels_db,
            CAMELS_FLOW_DIR = flow_dir,
            CAMELS_FORCING_DIR = forcing_dir,
            CAMELS_ATTR_DIR = attr_dir,
            CAMELS_ATTR_KEY_LST = attr_key_lst,
            CAMELS_GAUGE_FILE = gauge_id_file,
            CAMELS_NESTEDNESS_FILE=nestedness_information_file,
            CAMELS_BASINS_SHP = camels_shp_file,
            CAMELS_DOWNLOAD_URL_LST=download_url_lst,
        )

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
                "prcp",
                "pet",
                "discharge_vol",
            ]
        )

    def read_ystl_gage_flow_forcing(self, gage_id, t_range, var_type):
        """
        Read gage's streamflow or forcing from CAMELS-CH

        Parameters
        ----------
        gage_id
            the station id
        t_range
            the time range, for example, ["1981-01-01","2021-01-01"]
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
            "camels_ystl_" + gage_id + ".csv",
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
        read target values. for CAMELS-CH, they are streamflows.

        default target_cols is an one-value list
        Notice, the unit of target outputs in different regions are not totally same

        Parameters
        ----------
        gage_id_lst
            station ids
        t_range
            the time range, for example, ["1981-01-01","2021-01-01"]
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
                data_obs = self.read_ystl_gage_flow_forcing(
                    gage_id_lst[k], t_range, target_cols[j]
                )
                y[k, :, j] = data_obs
        # Keep unit of streamflow unified: we use ft3/s here
        # other units are m3/s -> ft3/s
        # y = self.unit_convert_streamflow_m3tofoot3(y)
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
            the time range, for example, ["1981-01-01","2021-01-01"]
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
        for j in tqdm(range(len(var_lst)), desc="Read forcing data of CAMELS-YSTL"):
            for k in tqdm(range(len(gage_id_lst))):
                data_forcing = self.read_ystl_gage_flow_forcing(
                    gage_id_lst[k], t_range, var_lst[j]
                )
                x[k, :, j] = data_forcing
        return x

    def get_attribute_units_dict(self):
        """

        Returns
        -------

        """
        # # delete the repetitive attribute item, "country".
        # duplicate_columns = attrs_df.columns[attrs_df.columns.duplicated()]
        # if duplicate_columns.size > 0:
        #     attrs_df = attrs_df.loc[:, ~attrs_df.columns.duplicated()]
        units_dict = {
            "area": "km^2",
        }
        return units_dict

    def cache_streamflow_xrdataset(self):
        """Save all basins' streamflow data in a netcdf file in the cache directory

        """
        filename_npy = "camels_" + self.region.lower() + "_" + self.forcing_type + "_streamflow.npy"
        filename_json = "camels_" + self.region.lower() + "_" + self.forcing_type + "_streamflow.json"
        cache_npy_file = CACHE_DIR.joinpath(filename_npy)
        json_file = CACHE_DIR.joinpath(filename_json)
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
