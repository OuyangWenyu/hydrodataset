"""
Author: Wenyu Ouyang
Date: 2023-07-18 11:45:25
LastEditTime: 2025-01-01 15:01:45
LastEditors: Wenyu Ouyang
Description: Test for caravan dataset reading
FilePath: \hydrodataset\tests\test_caravan.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import numpy as np
import pytest

from hydrodataset import ROOT_DIR
from hydrodataset.caravan import Caravan, _extract_unit


@pytest.fixture()
def caravan_us():
    return Caravan(
        os.path.join(ROOT_DIR, "caravan"),
        region="US",
    )
@pytest.fixture()
def caravan():
    return Caravan(
        os.path.join(ROOT_DIR, "caravan"),
        region="Global",
    )


def test_read_caravan_us(caravan_us):
    caravan_ids = caravan_us.read_object_ids()
    assert len(caravan_ids) == 671

    streamflow_types = caravan_us.get_target_cols()
    np.testing.assert_array_equal(streamflow_types, np.array(["streamflow"]))
    focing_types = caravan_us.get_relevant_cols()
    np.testing.assert_array_equal(
        focing_types[:3],
        np.array(
            [
                "dewpoint_temperature_2m_max",
                "dewpoint_temperature_2m_mean",
                "dewpoint_temperature_2m_min",
            ]
        ),
    )
    attr_types = caravan_us.get_constant_cols()
    np.testing.assert_array_equal(
        attr_types[:3],
        np.array(["aridity_ERA5_LAND", "aridity_FAO_PM", "frac_snow"]),
    )

    attrs = caravan_us.read_constant_cols(
        caravan_ids[:5],
        var_lst=["aridity_ERA5_LAND", "aridity_FAO_PM", "frac_snow"],
    )
    np.testing.assert_almost_equal(
        attrs,
        np.array(
            [
                [3.7639294, 0.4925567, 0.37453797],
                [4.539418, 0.5809156, 0.3354396],
                [4.2636347, 0.5328785, 0.31613037],
                [3.9475863, 0.51058507, 0.3062726],
                [3.6948392, 0.5015839, 0.2990631],
            ]
        ),
    )
    forcings = caravan_us.read_relevant_cols(
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
    flows = caravan_us.read_target_cols(
        caravan_ids[:5], ["1990-01-01", "2009-12-31"], target_cols=["streamflow"]
    )
    np.testing.assert_array_equal(
        flows.to_array().transpose("gauge_id", "date", "variable").shape,
        np.array([5, 7305, 1]),
    )


def all_elements_in_array(elements_list, np_array):
    return np.all(np.isin(elements_list, np_array))


def test_read_caravan(caravan):
    caravan_ids = caravan.read_object_ids()
    assert len(caravan_ids) == 15960

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
        ["p_mean", "pet_mean_ERA5_LAND", "aridity_ERA5_LAND"],
        attr_types,
    )

    caravan_ids_chosen = caravan_ids[:3].tolist() + caravan_ids[-2:].tolist()
    attrs = caravan.read_constant_cols(
        caravan_ids_chosen,
        var_lst=["p_mean", "pet_mean_ERA5_LAND", "aridity_ERA5_LAND"],
    )
    np.testing.assert_almost_equal(
        attrs,
        np.array(
            [
                [3.175454, 11.952184, 3.7639294],
                [3.2038372, 14.543556, 4.539418],
                [3.1840692, 13.575707, 4.2636347],
                [2.7544532, 3.6424763, 1.3223954],
                [3.0899684, 2.9222033, 0.9457065],
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


def test_cache_caravan(caravan):
    caravan.cache_xrdataset(checkregion=None)



def test_read_ts_xrdataset(caravan):
    caravan_ids = caravan.read_object_ids()
    ts_data = caravan.read_ts_xrdataset(
        caravan_ids[:3].tolist() + caravan_ids[-2:].tolist(),
        ["1990-01-01", "2009-12-31"],
        var_lst=None,
    )
    print("TS Data for Selected IDs:", ts_data)  # Print the time series data for debugging


def test_read_attr_xrdataset(caravan):
    caravan_ids = caravan.read_object_ids()
    attr_data = caravan.read_attr_xrdataset(
        caravan_ids[:3].tolist() + caravan_ids[-2:].tolist(),
        ["p_mean", "pet_mean_ERA5_LAND", "aridity_ERA5_LAND"],
    )
    print(attr_data)


def test_streamflow_unit(caravan):
    assert caravan.streamflow_unit == "mm/d"


def test_read_area(caravan):
    caravan_ids = caravan.read_object_ids()
    area = caravan.read_area(caravan_ids[:3].tolist() + caravan_ids[-2:].tolist())
    print(area)


def test_read_prcp_mean(caravan):
    # be careful for the unit!
    caravan_ids = caravan.read_object_ids()
    prcp_mean = caravan.read_mean_prcp(
        caravan_ids[:3].tolist() + caravan_ids[-2:].tolist()
    )
    print(prcp_mean)


def test_extract_unit():
    units_string = """
    snow_depth_water_equivalent: ERA5-Land Snow-Water-Equivalent [mm]
    surface_net_solar_radiation: Surface net solar radiation [W/m2]
    surface_net_thermal_radiation: Surface net thermal radiation [W/m2]
    surface_pressure: Surface pressure [kPa]
    temperature_2m: 2m air temperature [°C]
    u_component_of_wind_10m: U-component of wind at 10m [m/s]
    v_component_of_wind_10m: V-component of wind at 10m [m/s]
    volumetric_soil_water_layer_1: ERA5-Land volumetric soil water layer 1 (0-7cm) [m3/m3]
    volumetric_soil_water_layer_2: ERA5-Land volumetric soil water layer 2 (7-28cm) [m3/m3]
    volumetric_soil_water_layer_3: ERA5-Land volumetric soil water layer 3 (28-100cm) [m3/m3]
    volumetric_soil_water_layer_4: ERA5-Land volumetric soil water layer 4 (100-289cm) [m3/m3]
    total_precipitation: Total precipitation [mm]
    potential_evaporation: ERA5-Land Potential Evapotranspiration [mm]
    """

    # Test for streamflow
    assert _extract_unit("streamflow", units_string) == "mm"

    # Test for dewpoint_temperature_2m
    assert _extract_unit("dewpoint_temperature_2m", units_string) == "°C"

    # Test for temperature_2m
    assert _extract_unit("temperature_2m_mean", units_string) == "°C"

    # Test for unknown variable
    assert _extract_unit("unknown_variable", units_string) == "unknown"

    # Test for variable with no unit
    units_string_no_unit = """
    temperature_2m: [K]
    dewpoint_temperature_2m: [K]
    variable_no_unit:
    """
    assert _extract_unit("variable_no_unit", units_string_no_unit) == "unknown"
