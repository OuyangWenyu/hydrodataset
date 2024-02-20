"""
Author: Wenyu Ouyang
Date: 2022-09-05 23:20:24
LastEditTime: 2024-02-20 19:33:43
LastEditors: Wenyu Ouyang
Description: main modules for hydrodataset
FilePath: \hydrodataset\hydrodataset\hydro_dataset.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

from abc import ABC
from pathlib import Path
from typing import Union

import numpy as np

from hydrodataset import ROOT_DIR


class HydroDataset(ABC):
    """An interface for Hydrological Dataset

    For unit, we use Pint package's unit system -- unit registry

    Parameters
    ----------
    ABC : _type_
        _description_
    """

    def __init__(self, data_path):
        self.data_source_dir = Path(ROOT_DIR, data_path)
        if not self.data_source_dir.is_dir():
            self.data_source_dir.mkdir(parents=True)

    def get_name(self):
        raise NotImplementedError

    def set_data_source_describe(self):
        raise NotImplementedError

    def download_data_source(self):
        raise NotImplementedError

    def read_object_ids(self, object_params=None) -> np.array:
        raise NotImplementedError

    def read_target_cols(
        self, object_ids=None, t_range_list=None, target_cols=None, **kwargs
    ) -> np.array:
        raise NotImplementedError

    def read_relevant_cols(
        self, object_ids=None, t_range_list: list = None, relevant_cols=None, **kwargs
    ) -> Union[np.array, list]:
        """3d data (site_num * time_length * var_num), time-series data"""
        raise NotImplementedError

    def read_constant_cols(
        self, object_ids=None, constant_cols=None, **kwargs
    ) -> np.array:
        """2d data (site_num * var_num), non-time-series data"""
        raise NotImplementedError

    def read_other_cols(
        self, object_ids=None, other_cols: dict = None, **kwargs
    ) -> dict:
        """some data which cannot be easily treated as constant vars or time-series with same length as relevant vars
        CONVENTION: other_cols is a dict, where each item is also a dict with all params in it
        """
        raise NotImplementedError

    def get_constant_cols(self) -> np.array:
        """the constant cols in this data_source"""
        raise NotImplementedError

    def get_relevant_cols(self) -> np.array:
        """the relevant cols in this data_source"""
        raise NotImplementedError

    def get_target_cols(self) -> np.array:
        """the target cols in this data_source"""
        raise NotImplementedError

    def get_other_cols(self) -> dict:
        """the other cols in this data_source"""
        raise NotImplementedError

    def cache_xrdataset(self, **kwargs):
        """cache xarray dataset and pandas feather for faster reading"""
        raise NotImplementedError

    def read_ts_xrdataset(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs
    ):
        """read time-series xarray dataset"""
        raise NotImplementedError

    def read_attr_xrdataset(self, gage_id_lst=None, var_lst=None, **kwargs):
        """read attribute pandas feather"""
        raise NotImplementedError

    def read_area(self, gage_id_lst=None):
        """read area of each basin/unit"""
        raise NotImplementedError

    def read_mean_prcp(self, gage_id_lst=None):
        """read mean precipitation of each basin/unit"""
        raise NotImplementedError
