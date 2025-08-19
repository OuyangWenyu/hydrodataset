"""
Author: Wenyu Ouyang
Date: 2023-07-18 11:45:25
LastEditTime: 2025-01-06 10:23:09
LastEditors: Muhammad Adnan
Description: Test for caravan dataset reading
FilePath: r"\hydrodataset\tests\test_grdc_caravan.py"
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import numpy as np
import pandas as pd
import pytest
from hydrodataset.grdc_caravan import CACHE_DIR

from hydrodataset import ROOT_DIR
from hydrodataset.grdc_caravan import GrdcCaravan


@pytest.fixture()
def grdc_caravan():
    return GrdcCaravan(os.path.join(ROOT_DIR, "GRDC-Caravan"), download=False)


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
            [0.16364825, 0.07590102, 0.46380588],
            [0.16711121, 0.07631393, 0.45666553],
            [0.14861074, 0.07802555, 0.52503302],
            [0.89598562, 0.00450208, 0.00502473],
            [0.9214931, 0.00224583, 0.00243717],
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


# def test_cache_grdc_caravan(grdc_caravan):
#     grdc_caravan.cache_attributes_xrdataset()
#     grdc_caravan.cache_xrdataset



def test_cache_grdc_caravan(grdc_caravan):
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Call attribute caching
    grdc_caravan.cache_attributes_xrdataset()
    cache_attr_file = os.path.join(CACHE_DIR, "grdc_caravan_attributes.nc")
    assert os.path.isfile(cache_attr_file), "Attribute cache file was not created"

    # Call time series caching
    grdc_caravan.cache_timeseries_xrdataset()
    part_files = [f for f in os.listdir(CACHE_DIR) if f.startswith("grdc_caravan_timeseries_part_")]
    assert part_files, "Time series cache part files were not created"



def test_read_timeseries(grdc_caravan):
    caravan_ids = grdc_caravan.read_object_ids()
    t_range = ["1990-01-01", "2009-12-31"]
    var_lst = ["streamflow", "total_precipitation_sum"]

    # Test reading timeseries with default parameters
    # ts_data_default = grdc_caravan.read_timeseries(caravan_ids[:5])
    # assert ts_data_default.shape[0] == len(caravan_ids)
    # assert ts_data_default.shape[1] == len(pd.date_range("1980-01-01", "2023-12-31"))
    # assert ts_data_default.shape[2] == len(grdc_caravan.get_relevant_cols())

    # Test reading timeseries with specific basin_ids, t_range, and var_lst
    ts_data_specific = grdc_caravan._read_timeseries_data(
        gage_id_lst=caravan_ids[:5], t_range=t_range, var_lst=var_lst
    )
    assert len(ts_data_specific.gauge_id) == 5
    assert "date" in ts_data_specific.dims
    assert all(v in ts_data_specific.data_vars for v in ["streamflow", "total_precipitation_sum"])

    # Test reading timeseries with only basin_ids
    ts_data_basin_ids = grdc_caravan._read_timeseries_data(gage_id_lst=caravan_ids[:5])
    assert len(ts_data_specific.gauge_id) == 5
    assert str(ts_data_specific.date.values[0])[:10] == "1990-01-01"
    assert str(ts_data_specific.date.values[-1])[:10] == "2009-12-31"
    assert len(ts_data_basin_ids.data_vars) == len(grdc_caravan.get_relevant_cols())


def test_read_ts_xrdataset(grdc_caravan):
    ts_data = grdc_caravan.read_ts_xrdataset(
        ["GRDC_1197507"],
        ["1980-01-01", "2023-05-18"],
        var_lst=["streamflow"],
    )
    assert ts_data is not None, "read_ts_xrdataset returned None (no data found)"
    ts_data_1 = grdc_caravan.read_target_cols(
        ["GRDC_1197507"],
        ["1980-01-01", "2023-05-18"],
        target_cols=["streamflow"],
    )
    assert ts_data_1 is not None, "read_target_cols returned None (no data found)"
    np.testing.assert_almost_equal(
        ts_data.to_array().transpose("basin", "time", "variable").to_numpy(),
        ts_data_1.to_array().transpose("gauge_id", "date", "variable").to_numpy(),
        # tolerence we set to 1e-5
        decimal=5,
    )


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

def test_read_attributes(grdc_caravan):
    caravan_ids = grdc_caravan.read_object_ids()
    var_lst = ['p_mean', 'pet_mean', 'area']  # replace with actual variables in your dataset
    attributes = grdc_caravan._read_attr_files_grdc(
        caravan_ids[:3].tolist() + caravan_ids[-2:].tolist(),
        var_lst
    )
    print(attributes)