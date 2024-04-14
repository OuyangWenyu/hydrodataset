"""
Author: Wenyu Ouyang
Date: 2024-04-14 14:19:49
LastEditTime: 2024-04-14 14:44:42
LastEditors: Wenyu Ouyang
Description: some util functions for data processing
FilePath: \hydrodataset\hydrodataset\hds_utils.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import numpy as np
from typing import Union

from hydrodataset.camels import Camels
from hydrodataset.multi_datasets import MultiDatasets


def usgs_screen_streamflow(
    gages: Union[Camels, MultiDatasets],
    usgs_ids: list,
    time_range: list,
    **kwargs,
) -> list:
    """
    according to the criteria and its ancillary condition--thresh of streamflow data,
    choose appropriate ones from the given usgs sites

    Parameters
    ----------
    gages
        Camels, CamelsSeries object
    usgs_ids: list
        given sites' ids
    time_range: list
        chosen time range
    flow_type
        flow's name in data file; default is usgsFlow for CAMELS-US
    kwargs
        all criteria

    Returns
    -------
    list
        sites_chosen: [] -- ids of chosen gages

    Examples
    --------
        >>> usgs_screen_streamflow(gages, ["02349000","08168797"], ["1995-01-01","2015-01-01"], **{'missing_data_ratio': 0, 'zero_value_ratio': 1})
    """
    usgs_values = gages.read_target_cols(usgs_ids, time_range, gages.get_target_cols())[
        :, :, 0
    ]
    sites_index = np.arange(usgs_values.shape[0])
    sites_chosen = np.ones(usgs_values.shape[0])
    for i in range(sites_index.size):
        # loop for every site
        runoff = usgs_values[i, :]
        for criteria, thresh in kwargs.items():
            # if any criteria is not matched, we can filter this site
            if sites_chosen[sites_index[i]] == 0:
                break
            if criteria == "missing_data_ratio":
                nan_length = runoff[np.isnan(runoff)].size
                sites_chosen[sites_index[i]] = (
                    0 if nan_length / runoff.size > thresh else 1
                )
            elif criteria == "zero_value_ratio":
                zero_length = runoff.size - np.count_nonzero(runoff)
                thresh = kwargs[criteria]
                sites_chosen[sites_index[i]] = (
                    0 if zero_length / runoff.size > thresh else 1
                )
            else:
                print(
                    "Oops! That is not valid value. Try missing_data_ratio or zero_value_ratio ..."
                )
    return [usgs_ids[i] for i in range(len(sites_chosen)) if sites_chosen[i] > 0]
