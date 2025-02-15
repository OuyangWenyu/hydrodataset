"""
Author: Wenyu Ouyang
Date: 2022-09-06 23:42:46
LastEditTime: 2024-11-11 16:44:31
LastEditors: Wenyu Ouyang
Description: examples for using hydrodataset
FilePath: \hydrodataset\examples\scripts.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import os
import numpy as np

from hydrodataset import Camels
from hydrodataset import CamelsCh

camels_aus_path = os.path.join("camels", "camels_aus")
camels_br_path = os.path.join("camels", "camels_br")
camels_cl_path = os.path.join("camels", "camels_cl")
camels_gb_path = os.path.join("camels", "camels_gb")
camels_us_path = os.path.join("camels", "camels_us")
camels_aus_v2_path = os.path.join("camels", "camels_aus_v2")
camels_ch_path = os.path.join("camels", "camels_ch")

aus_v2_region = "AUS_v2"
aus_region = "AUS"
br_region = "BR"
cl_region = "CL"
gb_region = "GB"
us_region = "US"
ch_region = "CH"

# ------------------------------ US --------------------------------
# if files is not zipped, set download=True to unzip the files
camels_us = Camels(camels_us_path, download=False, region=us_region)
gage_ids = camels_us.read_object_ids()
flows = camels_us.read_target_cols(
    gage_ids[:5], ["2013-01-01", "2014-01-01"], target_cols=["usgsFlow"]
)
print(flows)

gage_ids = camels_us.read_object_ids()
assert gage_ids.size == 671
attrs = camels_us.read_constant_cols(
    gage_ids[:5], var_lst=["soil_conductivity", "elev_mean", "geol_1st_class"]
)
np.testing.assert_almost_equal(
    attrs,
    np.array(
        [
            [1.10652248, 250.31, 10.0],
            [2.37500506, 92.68, 0.0],
            [1.28980735, 143.8, 10.0],
            [1.37329168, 247.8, 10.0],
            [2.61515428, 310.38, 7.0],
        ]
    ),
)
forcings = camels_us.read_relevant_cols(
    gage_ids[:5], ["1990-01-01", "2010-01-01"], var_lst=["dayl", "prcp", "srad"]
)
np.testing.assert_array_equal(forcings.shape, np.array([5, 7305, 3]))
flows = camels_us.read_target_cols(
    gage_ids[:5], ["1990-01-01", "2010-01-01"], target_cols=["usgsFlow"]
)
np.testing.assert_array_equal(flows.shape, np.array([5, 7305, 1]))
streamflow_types = camels_us.get_target_cols()
np.testing.assert_array_equal(streamflow_types, np.array(["usgsFlow"]))
focing_types = camels_us.get_relevant_cols()
np.testing.assert_array_equal(
    focing_types, np.array(["dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"])
)
attr_types = camels_us.get_constant_cols()
np.testing.assert_array_equal(
    attr_types[:3], np.array(["gauge_lat", "gauge_lon", "elev_mean"])
)

# we highly recommend to cache the xrdataset for faster access
camels_us.cache_xrdataset()
forcings = camels_us.read_ts_xrdataset(
    gage_id_lst=gage_ids[:5],
    t_range=["2013-01-01", "2014-01-01"],
    var_lst=["prcp", "srad", "tmax"],
)
flows = camels_us.read_ts_xrdataset(
    gage_id_lst=gage_ids[:5],
    t_range=["2013-01-01", "2014-01-01"],
    # NOTE: the variable name is "streamflow" instead of "usgsFlow" after caching
    var_lst=["streamflow"],
)
attrs = camels_us.read_attr_xrdataset(
    gage_id_lst=gage_ids[:5],
    var_lst=["soil_conductivity", "elev_mean", "geol_1st_class"],
)

# ------------------------------ AUS --------------------------------
camels_aus = Camels(camels_aus_path, download=False, region=aus_region)
gage_ids = camels_aus.read_object_ids()
assert gage_ids.size == 222
attrs = camels_aus.read_constant_cols(
    gage_ids[:5], var_lst=["catchment_area", "slope_fdc", "geol_sec"]
)
np.testing.assert_array_equal(
    attrs,
    np.array(
        [
            [1.25773e04, 3.66793e-01, 6.00000e00],
            [1.13929e04, 3.29998e-01, 6.00000e00],
            [5.65300e02, 1.78540e-02, 6.00000e00],
            [4.58300e02, 5.00234e-01, 6.00000e00],
            [7.73170e03, 3.74751e-01, 1.00000e00],
        ]
    ),
)
forcings = camels_aus.read_relevant_cols(
    gage_ids[:5],
    ["1990-01-01", "2010-01-01"],
    # For the AUS region, there are two forcing types,
    # but we didn't specify the forcing type in forcing_type argument,
    # because it is more convenient to use the forcing_name directly.
    # As one can see, all the forcing names have a postfix
    var_lst=["precipitation_AWAP", "et_morton_actual_SILO", "tmin_SILO"],
)
np.testing.assert_array_equal(forcings.shape, np.array([5, 7305, 3]))
flows = camels_aus.read_target_cols(
    gage_ids[:5],
    ["1990-01-01", "2010-01-01"],
    target_cols=["streamflow_MLd", "streamflow_mmd"],
)
np.testing.assert_array_equal(flows.shape, np.array([5, 7305, 2]))
streamflow_types = camels_aus.get_target_cols()
np.testing.assert_array_equal(
    streamflow_types,
    np.array(
        [
            "streamflow_MLd",
            "streamflow_MLd_inclInfilled",
            "streamflow_mmd",
            "streamflow_QualityCodes",
        ]
    ),
)
focing_types = camels_aus.get_relevant_cols()
np.testing.assert_array_equal(
    np.sort(focing_types),
    np.sort(
        [
            "precipitation_AWAP",
            "precipitation_SILO",
            "precipitation_var_AWAP",
            "et_morton_actual_SILO",
            "et_morton_point_SILO",
            "et_morton_wet_SILO",
            "et_short_crop_SILO",
            "et_tall_crop_SILO",
            "evap_morton_lake_SILO",
            "evap_pan_SILO",
            "evap_syn_SILO",
            "solarrad_AWAP",
            "tmax_AWAP",
            "tmin_AWAP",
            "vprp_AWAP",
            "mslp_SILO",
            "radiation_SILO",
            "rh_tmax_SILO",
            "rh_tmin_SILO",
            "tmax_SILO",
            "tmin_SILO",
            "vp_deficit_SILO",
            "vp_SILO",
        ]
    ),
)
attr_types = camels_aus.get_constant_cols()
np.testing.assert_array_equal(
    attr_types[:3], np.array(["station_name", "drainage_division", "river_region"])
)

# # ---------------------------- AUS-V2 -------------------------------
camels_aus_v2 = Camels(camels_aus_v2_path, download=False, region=aus_v2_region)
gage_ids = camels_aus_v2.read_object_ids()
assert gage_ids.size == 561
attrs = camels_aus_v2.read_constant_cols(
    gage_ids[:5], var_lst=["catchment_area", "geol_sec", "metamorph"]
)
print(attrs)
forcings = camels_aus_v2.read_relevant_cols(
    gage_ids[:5],
    ["1990-01-01", "2010-01-01"],
    var_lst=["precipitation_AGCD", "et_morton_actual_SILO", "tmin_SILO"],
)
print(forcings.shape)
flows = camels_aus_v2.read_target_cols(
    gage_ids[:5],
    ["1990-01-01", "2010-01-01"],
    target_cols=["streamflow_MLd", "streamflow_mmd"],
)
print(flows.shape)
streamflow_types = camels_aus_v2.get_target_cols()
print(streamflow_types)
focing_types = camels_aus_v2.get_relevant_cols()
print(focing_types)
attr_types = camels_aus_v2.get_constant_cols()
print(attr_types)

# ------------------------------ BR --------------------------------
camels_br = Camels(camels_br_path, download=False, region=br_region)
gage_ids = camels_br.read_object_ids()
assert gage_ids.size == 897
attrs = camels_br.read_constant_cols(
    gage_ids[:5], var_lst=["geol_class_1st", "p_mean", "runoff_ratio"]
)
np.testing.assert_array_equal(
    attrs,
    np.array(
        [
            [8.0, 6.51179, 0.55336],
            [6.0, 5.38941, 0.72594],
            [6.0, 5.70191, 0.759],
            [6.0, 5.19877, 0.39463],
            [8.0, 5.49805, 0.38579],
        ]
    ),
)
forcings = camels_br.read_relevant_cols(
    gage_ids[:5],
    ["1995-01-01", "2015-01-01"],
    var_lst=[
        "precipitation_chirps",
        "evapotransp_gleam",
        "potential_evapotransp_gleam",
        "temperature_min_cpc",
    ],
)
np.testing.assert_array_equal(forcings.shape, np.array([5, 7305, 4]))
# start from 1995/01/01 end with 2017/04/30
flows = camels_br.read_target_cols(
    gage_ids[:5],
    ["1995-01-01", "2015-01-01"],
    target_cols=["streamflow_m3s", "streamflow_mm_selected_catchments"],
)
np.testing.assert_array_equal(flows.shape, np.array([5, 7305, 2]))
streamflow_types = camels_br.get_target_cols()
np.testing.assert_array_equal(
    streamflow_types,
    np.array(
        [
            "streamflow_m3s",
            "streamflow_mm_selected_catchments",
            "streamflow_simulated",
        ]
    ),
)
focing_types = camels_br.get_relevant_cols()
np.testing.assert_array_equal(
    focing_types,
    np.array(
        [
            "precipitation_chirps",
            "precipitation_mswep",
            "precipitation_cpc",
            "evapotransp_gleam",
            "evapotransp_mgb",
            "potential_evapotransp_gleam",
            "temperature_min_cpc",
            "temperature_mean_cpc",
            "temperature_max_cpc",
        ]
    ),
)
attr_types = camels_br.get_constant_cols()
np.testing.assert_array_equal(
    attr_types[:3], np.array(["p_mean", "pet_mean", "et_mean"])
)

# ------------------------------ CL --------------------------------
camels_cl = Camels(camels_cl_path, download=False, region=cl_region)
gage_ids = camels_cl.read_object_ids()
assert gage_ids.size == 516
attrs = camels_cl.read_constant_cols(
    gage_ids[:5], var_lst=["geol_class_1st", "crop_frac"]
)
np.testing.assert_almost_equal(
    attrs,
    np.array(
        [
            [9.0, 0.0],
            [9.0, 0.014243],
            [9.0, 0.020827],
            [10.0, 0.1055],
            [10.0, 0.0684],
        ]
    ),
    decimal=4,
)
forcings = camels_cl.read_relevant_cols(
    gage_ids[:5],
    ["1995-01-01", "2015-01-01"],
    var_lst=["pet_8d_modis", "precip_cr2met", "swe"],
)
np.testing.assert_array_equal(forcings.shape, np.array([5, 7305, 3]))
flows = camels_cl.read_target_cols(
    gage_ids[:5],
    ["1995-01-01", "2015-01-01"],
    target_cols=["streamflow_m3s", "streamflow_mm"],
)
np.testing.assert_array_equal(flows.shape, np.array([5, 7305, 2]))
streamflow_types = camels_cl.get_target_cols()
np.testing.assert_array_equal(
    streamflow_types, np.array(["streamflow_m3s", "streamflow_mm"])
)
focing_types = camels_cl.get_relevant_cols()
np.testing.assert_array_equal(
    focing_types,
    np.array(
        [
            "precip_cr2met",
            "precip_chirps",
            "precip_mswep",
            "precip_tmpa",
            "tmin_cr2met",
            "tmax_cr2met",
            "tmean_cr2met",
            "pet_8d_modis",
            "pet_hargreaves",
            "swe",
        ]
    ),
)
attr_types = camels_cl.get_constant_cols()
np.testing.assert_array_equal(
    attr_types[:3], np.array(["gauge_name", "gauge_lat", "gauge_lon"])
)

# ------------------------------ GB --------------------------------
camels_gb = Camels(camels_gb_path, download=False, region=gb_region)
gage_ids = camels_gb.read_object_ids()
assert gage_ids.size == 671
attrs = camels_gb.read_constant_cols(
    gage_ids[:5], var_lst=["p_mean", "slope_fdc", "gauge_name"]
)
np.testing.assert_array_equal(
    attrs,
    np.array(
        [
            [2.29, 1.94, 596.0],
            [2.31, 1.95, 670.0],
            [2.65, 4.01, 647.0],
            [2.31, 1.54, 393.0],
            [2.29, 1.47, 217.0],
        ]
    ),
)
forcings = camels_gb.read_relevant_cols(
    gage_ids[:5],
    ["1995-01-01", "2015-01-01"],
    var_lst=["precipitation", "pet", "temperature", "peti"],
)
np.testing.assert_array_equal(forcings.shape, np.array([5, 7305, 4]))
flows = camels_gb.read_target_cols(
    gage_ids[:5],
    ["1995-01-01", "2015-01-01"],
    target_cols=["discharge_spec", "discharge_vol"],
)
np.testing.assert_array_equal(flows.shape, np.array([5, 7305, 2]))
streamflow_types = camels_gb.get_target_cols()
np.testing.assert_array_equal(
    streamflow_types, np.array(["discharge_spec", "discharge_vol"])
)
focing_types = camels_gb.get_relevant_cols()
np.testing.assert_array_equal(
    focing_types,
    np.array(
        [
            "precipitation",
            "pet",
            "temperature",
            "peti",
            "humidity",
            "shortwave_rad",
            "longwave_rad",
            "windspeed",
        ]
    ),
)
attr_types = camels_gb.get_constant_cols()
np.testing.assert_array_equal(
    attr_types[:3], np.array(["p_mean", "pet_mean", "aridity"])
)



camelsch = CamelsCh()
gage_ids = camelsch.read_object_ids()
assert gage_ids.size == 331
attrs = camelsch.read_constant_cols(
    gage_ids[:5],["area","p_mean","crop_perc"]
)
print(attrs)
forcings = camelsch.read_relevant_cols(
        gage_ids[:5],
        ["1981-01-01","2020-12-31"],
        var_lst=["waterlevel(m)", "precipitation(mm/d)", "temperature_min(degC)", "temperature_mean(degC)", "temperature_max(degC)", "rel_sun_dur(%)", "swe(mm)"]
    )
print(forcings)
streamflow = camelsch.read_target_cols(
        gage_ids[:5],
        ["1981-01-01","2020-12-31"],
        target_cols=["discharge_vol(m3/s)", "discharge_spec(mm/d)"],
    )
print(streamflow)
attrs_types = camelsch.read_constant_cols()
print(attrs_types)
forcing_types = camelsch.read_relevant_cols()
print(forcing_types)
streamflow_types = camelsch.read_target_cols()
print(streamflow_types)
