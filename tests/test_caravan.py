"""
Author: Wenyu Ouyang
Date: 2023-07-18 11:45:25
LastEditTime: 2023-10-17 09:38:01
LastEditors: Wenyu Ouyang
Description: Test for caravan dataset reading
FilePath: /hydrodataset/tests/test_caravan.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import os
import numpy as np

from hydrodataset import ROOT_DIR
from hydrodataset.caravan import Caravan


def test_read_caravan_us():
    caravan = Caravan(
        os.path.join(ROOT_DIR, "caravan"),
        region="US",
    )
    caravan_ids = caravan.read_object_ids()
    assert len(caravan_ids) == 482

    streamflow_types = caravan.get_target_cols()
    np.testing.assert_array_equal(streamflow_types, np.array(["streamflow"]))
    focing_types = caravan.get_relevant_cols()
    np.testing.assert_array_equal(
        focing_types[:3],
        np.array(
            [
                "snow_depth_water_equivalent_mean",
                "surface_net_solar_radiation_mean",
                "surface_net_thermal_radiation_mean",
            ]
        ),
    )
    attr_types = caravan.get_constant_cols()
    np.testing.assert_array_equal(
        attr_types[:3],
        np.array(["p_mean", "pet_mean", "aridity"]),
    )

    attrs = caravan.read_constant_cols(
        caravan_ids[:5],
        var_lst=["p_mean", "pet_mean", "aridity"],
    )
    np.testing.assert_almost_equal(
        attrs,
        np.array(
            [
                [3.20371278, 14.54395441, 4.53971857],
                [3.2891471, 12.98397837, 3.94752134],
                [3.2756958, 12.1030139, 3.69479177],
                [3.40199329, 10.64317544, 3.12851159],
                [3.66423575, 10.99960641, 3.00188284],
            ]
        ),
    )
    forcings = caravan.read_relevant_cols(
        caravan_ids[:5],
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
    flows = caravan.read_target_cols(
        caravan_ids[:5], ["1990-01-01", "2009-12-31"], target_cols=["streamflow"]
    )
    np.testing.assert_array_equal(
        flows.to_array().transpose("gauge_id", "date", "variable").shape,
        np.array([5, 7305, 1]),
    )


def all_elements_in_array(elements_list, np_array):
    return np.all(np.isin(elements_list, np_array))


def test_read_caravan():
    caravan = Caravan(
        os.path.join(ROOT_DIR, "caravan"),
        region="Global",
    )
    caravan_ids = caravan.read_object_ids()
    assert len(caravan_ids) == 6830

    streamflow_types = caravan.get_target_cols()
    np.testing.assert_array_equal(streamflow_types, np.array(["streamflow"]))
    focing_types = caravan.get_relevant_cols()
    assert all_elements_in_array(
        [
            "snow_depth_water_equivalent_mean",
            "surface_net_solar_radiation_mean",
            "surface_net_thermal_radiation_mean",
        ],
        focing_types,
    )
    attr_types = caravan.get_constant_cols()
    assert all_elements_in_array(
        ["p_mean", "pet_mean", "aridity"],
        attr_types,
    )

    caravan_ids_chosen = caravan_ids[:3].tolist() + caravan_ids[-2:].tolist()
    attrs = caravan.read_constant_cols(
        caravan_ids_chosen,
        var_lst=["p_mean", "pet_mean", "aridity"],
    )
    np.testing.assert_almost_equal(
        attrs,
        np.array(
            [
                [3.20371278, 14.54395441, 4.53971857],
                [3.2891471, 12.98397837, 3.94752134],
                [3.2756958, 12.1030139, 3.69479177],
                [3.00678349, 3.6481265, 1.2132987],
                [2.75425491, 3.64257786, 1.3225275],
            ]
        ),
    )
    forcings = caravan.read_relevant_cols(
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
    flows = caravan.read_target_cols(
        caravan_ids_chosen, ["1990-01-01", "2009-12-31"], target_cols=["streamflow"]
    )
    np.testing.assert_array_equal(
        flows.to_array().transpose("gauge_id", "date", "variable").shape,
        np.array([5, 7305, 1]),
    )


def test_cache_caravan():
    caravan = Caravan(
        os.path.join(ROOT_DIR, "caravan"),
        region="Global",
    )
    caravan.cache_xrdataset()
