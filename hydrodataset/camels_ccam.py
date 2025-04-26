import os
import logging
import collections
import pandas as pd
import numpy as np
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

camelsccam_arg = {
    "forcing_type": "observation",
    "gauge_id_tag": "basin_id",
    "area_tag": ["area", ],
    "meanprcp_unit_tag": [["pre_mean"], "mm/d"],
    "time_range": {
        "observation": ["1990-01-01","2021-04-01"],
    },
    "target_cols": ["q", ],
    "b_nestedness": False,
    "forcing_unit": ["m", "mm/day", "°C", "°C", "°C", "%", "mm"],
    "data_file_attr": {
        "sep": ",",
        "header": 1,
        "attr_file_str": ["2_", "_attributes.csv", "_attributes_obs.csv", ]
    },
}

class CamelsCcam(Camels):
    def __init__(
        self,
        data_path = os.path.join("camels","camels_ccam"),
        download = False,
        region: str = "CCAM",
        arg: dict = camelsccam_arg,
    ):
        """
        Initialization for CAMELS-CCAM dataset

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
            the default is CAMELS-CCAM
        """
        super().__init__(data_path, download, region, arg)

    def _set_data_source_camels_describe(self, camels_db):
        # shp file of basins
        camels_shp_file = camels_db.joinpath(
            "catchment_delineations",
            "CAMELS_CCAM_sub_catchments.shp",
        )
        # flow and forcing data are in a same file
        flow_dir = camels_db.joinpath(
            "timeseries",
            "observation_based",
        )
        forcing_dir = flow_dir
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
            "catchment",
        ]
        gauge_id_file = attr_dir.joinpath("CAMELS_CCAM_hydrology_attributes_obs.csv")
        nestedness_information_file = None
        base_url = "https://zenodo.org/records/5729444"
        download_url_lst = [
            f"{base_url}/files/0_catchment_boundary.zip",
            f"{base_url}/files/1_meteorological.zip",
            f"{base_url}/files/2_location_topography.txt",
            f"{base_url}/files/3_land_cover_characteristics.txt",
            f"{base_url}/files/4_geological_characteristics.txt",
            f"{base_url}/files/5_climate_indices.txt",
            f"{base_url}/files/6_soil.txt",
            f"{base_url}/files/7_HydroMLYR.zip",
            f"{base_url}/files/8_attribute_descriptions.xlsx",
            f"{base_url}/files/9_code_data.zip",
            f"{base_url}/files/readme.txt",
        ]

        return collections.OrderedDict(
            CAMELS_DIR=camels_db,
            CAMELS_FLOW_DIR=flow_dir,
            CAMELS_FORCING_DIR=forcing_dir,
            CAMELS_ATTR_DIR=attr_dir,
            CAMELS_ATTR_KEY_LST=attr_key_lst,
            CAMELS_GAUGE_FILE=gauge_id_file,
            CAMELS_NESTEDNESS_FILE=nestedness_information_file,
            CAMELS_BASINS_SHP=camels_shp_file,
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
            data_folder, "2_","_attributes_obs.csv",","
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

    def read_ccam_gage_flow_forcing(self, gage_id, t_range, var_type):
        """
        Read gage's streamflow or forcing from CAMELS-CCAM

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
            "CAMELS_CCAM_obs_based_" + gage_id + ".csv",
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
        read target values. for CAMELS-CCAM, they are streamflows.

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
            For CAMELS-CCAM, it's ["q"]
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
            range(len(target_cols)), desc="Read streamflow data of CAMELS-CCAM"
        ):
            for k in tqdm(range(len(gage_id_lst))):
                data_obs = self.read_ch_gage_flow_forcing(
                    gage_id_lst[k], t_range, target_cols[j]
                )
                y[k, :, j] = data_obs
        # Keep unit of streamflow unified: we use ft3/s here
        # other units are m3/s -> ft3/s
        y = self.unit_convert_streamflow_m3tofoot3(y)
        return y
