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
    "gauge_id_tag": "gauge_id",
    "area_tag": ["area", ],
    "meanprcp_unit_tag": [["p_mean"], "mm/d"],
    "time_range": {
        "observation": ["1981-01-01","2021-01-01"],
    },
    "target_cols": ["discharge_vol(m3/s)", "discharge_spec(mm/d)"],
    "b_nestedness": False,
    "forcing_unit": ["m", "mm/day", "°C", "°C", "°C", "%", "mm"],
    "data_file_attr": {
        "sep": ",",
        "header": 1,
        "attr_file_str": ["CAMELS_CH_", "_attributes.csv", "_attributes_obs.csv", ]
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
        base_url = "https://zenodo.org/records/15025258"
        download_url_lst = [
            f"{base_url}/files/camels_ccam.zip",
            f"{base_url}/files/camels_ccam_data_description.pdf",
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
