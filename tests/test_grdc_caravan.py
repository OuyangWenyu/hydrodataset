"""
Author: Wenyu Ouyang
Date: 2023-07-18 11:45:25
LastEditTime: 2025-01-02 10:08:45
LastEditors: Wenyu Ouyang
Description: Test for caravan dataset reading
FilePath: /hydrodataset/tests/test_grdc_caravan.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import numpy as np
import pytest

from hydrodataset import ROOT_DIR
from hydrodataset.grdc_caravan import GrdcCaravan


@pytest.fixture()
def grdc_caravan():
    return GrdcCaravan(os.path.join(ROOT_DIR, "GRDC-Caravan"), download=True)


def all_elements_in_array(elements_list, np_array):
    return np.all(np.isin(elements_list, np_array))


def test_read_grdc_caravan(grdc_caravan):
    caravan_ids = grdc_caravan.read_object_ids()
    assert len(caravan_ids) == 5357

    streamflow_types = grdc_caravan.get_target_cols()
    np.testing.assert_array_equal(streamflow_types, np.array(["streamflow"]))
    focing_types = grdc_caravan.get_relevant_cols()
    assert all_elements_in_array(
        [
            "snow_depth_water_equivalent_mean",
            "surface_net_solar_radiation_mean",
            "surface_net_thermal_radiation_mean",
        ],
        focing_types,
    )
    attr_types = grdc_caravan.get_constant_cols()
    assert all_elements_in_array(
        ["p_mean", "pet_mean", "aridity"],
        attr_types,
    )

    caravan_ids_chosen = caravan_ids[:3].tolist() + caravan_ids[-2:].tolist()
    attrs = grdc_caravan.read_constant_cols(
        caravan_ids_chosen,
        var_lst=["p_mean", "pet_mean", "aridity"],
    )
    np.testing.assert_almost_equal(
        attrs,
        np.array(
            [
                [1.21353567, 7.50012016, 6.18038709],
                [1.23786902, 7.52147532, 6.07614793],
                [1.10887647, 7.50398445, 6.76719606],
                [3.91461539, 5.78396797, 1.47753161],
                [4.6179657, 1.45969522, 0.31609053],
            ]
        ),
    )
    forcings = grdc_caravan.read_relevant_cols(
        caravan_ids_chosen,
        ["1990-01-01", "2009-12-31"],
        var_lst=[
            "snow_depth_water_equivalent_mean",
            "surface_net_solar_radiation_mean",
            "surface_net_thermal_radiation_mean",
        ],
    )
    np.testing.assert_array_equal(
        forcings.to_array().transpose("gauge_id", "date", "variable").shape,
        np.array([5, 7305, 3]),
    )
    flows = grdc_caravan.read_target_cols(
        caravan_ids_chosen, ["1990-01-01", "2009-12-31"], target_cols=["streamflow"]
    )
    np.testing.assert_array_equal(
        flows.to_array().transpose("gauge_id", "date", "variable").shape,
        np.array([5, 7305, 1]),
    )


def test_cache_grdc_caravan(grdc_caravan):
    grdc_caravan.cache_attributes_xrdataset()
    grdc_caravan.cache_timeseries_xrdataset(checkregion=None)


def test_read_ts_xrdataset(grdc_caravan):
    caravan_ids = grdc_caravan.read_object_ids()
    ts_data = grdc_caravan.read_ts_xrdataset(
        caravan_ids[:3].tolist() + caravan_ids[-2:].tolist(),
        ["1990-01-01", "2009-12-31"],
        var_lst=None,
    )
    print(ts_data)


def test_read_attr_xrdataset(grdc_caravan):
    caravan_ids = grdc_caravan.read_object_ids()
    attr_data = grdc_caravan.read_attr_xrdataset(
        caravan_ids[:3].tolist() + caravan_ids[-2:].tolist(),
        ["p_mean", "pet_mean", "aridity"],
    )
    print(attr_data)


def test_streamflow_unit(grdc_caravan):
    assert grdc_caravan.streamflow_unit == "mm/d"


def test_read_area(grdc_caravan):
    caravan_ids = grdc_caravan.read_object_ids()
    area = grdc_caravan.read_area(caravan_ids[:3].tolist() + caravan_ids[-2:].tolist())
    print(area)


def test_read_prcp_mean(grdc_caravan):
    # be careful for the unit!
    caravan_ids = grdc_caravan.read_object_ids()
    prcp_mean = grdc_caravan.read_mean_prcp(
        caravan_ids[:3].tolist() + caravan_ids[-2:].tolist()
    )
    print(prcp_mean)
