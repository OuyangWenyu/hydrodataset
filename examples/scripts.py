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
from hydrodataset import CamelsDe
from hydrodataset import CamelsDk
from hydrodataset import CamelsSe
from hydrodataset import CamelsFr
from hydrodataset import CamelsInd

camels_aus_path = os.path.join("camels", "camels_aus")
camels_br_path = os.path.join("camels", "camels_br")
camels_cl_path = os.path.join("camels", "camels_cl")
camels_gb_path = os.path.join("camels", "camels_gb")
camels_us_path = os.path.join("camels", "camels_us")
camels_aus_v2_path = os.path.join("camels", "camels_aus_v2")
camels_ch_path = os.path.join("camels", "camels_ch")
camels_de_path = os.path.join("camels", "camels_de")
camels_dk_path = os.path.join("camels", "camels_dk")
camels_se_path = os.path.join("camels", "camels_se")
camels_fr_path = os.path.join("camels", "camels_fr")
camels_ind_path = os.path.join("camels", "camels_ind")

aus_v2_region = "AUS_v2"
aus_region = "AUS"
br_region = "BR"
cl_region = "CL"
gb_region = "GB"
us_region = "US"
ch_region = "CH"
de_region = "DE"
dk_region = "DK"
SE_region = "SE"
FR_region = "FR"
Ind_region = "IND"

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
        var_lst=["waterlevel(m)", "precipitation(mm/d)", "temperature_min(degC)",
                 "temperature_mean(degC)", "temperature_max(degC)", "rel_sun_dur(%)", "swe(mm)"]
    )
print(forcings)
streamflow = camelsch.read_target_cols(
        gage_ids[:5],
        ["1981-01-01","2020-12-31"],
        target_cols=["discharge_vol(m3/s)", "discharge_spec(mm/d)"],
    )
print(streamflow)
attrs_types = camelsch.get_constant_cols()
print(attrs_types)
forcing_types = camelsch.get_relevant_cols()
print(forcing_types)
streamflow_types = camelsch.get_target_cols()
print(streamflow_types)





camelsde = CamelsDe()
gage_ids = camelsde.read_object_ids()
assert gage_ids.size == 1555
attrs = camelsde.read_constant_cols(
    gage_ids[:5], ["area", "p_mean", "dams_num"]
)
print(attrs)
forcings = camelsde.read_relevant_cols(
    gage_ids[:5],
    ["1951-01-01", "2020-12-31"],
    var_lst = ["water_level", "precipitation_mean", "precipitation_min", "precipitation_median", "precipitation_max",
           "precipitation_stdev", "humidity_mean", "humidity_min", "humidity_median"]
)
print(forcings)
streamflow = camelsde.read_target_cols(
    gage_ids[:5],
    ["1951-01-01", "2020-12-31"],
    target_cols = ["discharge_vol", "discharge_spec"],
)
print(streamflow)
attrs_types = camelsde.get_constant_cols()
print(attrs_types)
forcing_types = camelsde.get_relevant_cols()
print(forcing_types)
streamflow_types = camelsde.get_target_cols()
print(streamflow_types)



camelsdk = CamelsDk()
gage_ids = camelsdk.read_object_ids()
assert gage_ids.size == 3333
attrs = camelsdk.read_constant_cols(
    gage_ids[:5], ["catch_area", "p_mean", "slope_median"]
)
print(attrs)
forcings = camelsdk.read_relevant_cols(
    gage_ids[:5],
    ["1989-01-02", "2023-12-31"],
    var_lst = ["precipitation","temperature","pet","DKM_dtp","DKM_eta",
               "DKM_wcr","DKM_sdr","DKM_sre","DKM_gwh","Qdkm","DKM_irr","Abstraction"]
)
print(forcings)
streamflow = camelsdk.read_target_cols(
    gage_ids[:5],
    ["1989-01-02", "2023-12-31"],
    target_cols = ["Qobs"],
)
print(streamflow)
attrs_types = camelsdk.get_constant_cols()
print(attrs_types)
forcing_types = camelsdk.get_relevant_cols()
print(forcing_types)
streamflow_types = camelsdk.get_target_cols()
print(streamflow_types)




camelsse = CamelsSe()
gage_ids = camelsse.read_object_ids()
assert gage_ids.size == 50
attrs = camelsse.read_constant_cols(
    gage_ids[:5], ["Area_km2", "Pmean_mm_year", "Wetlands_percentage"]
)
print(attrs)
forcings = camelsse.read_relevant_cols(
    gage_ids[:5],
    ["1961-01-01", "2020-12-31"],
    var_lst = ["Pobs_mm", "Tobs_C"]
)
print(forcings)
streamflow = camelsse.read_target_cols(
    gage_ids[:5],
    ["1961-01-01", "2020-12-31"],
    target_cols = ["Qobs_m3s", "Qobs_mm"],
)
print(streamflow)
attrs_types = camelsse.get_constant_cols()
print(attrs_types)
forcing_types = camelsse.get_relevant_cols()
print(forcing_types)
streamflow_types = camelsse.get_target_cols()
print(streamflow_types)




camelsfr = CamelsFr()
gage_ids = camelsfr.read_object_ids()
assert gage_ids.size == 654
attrs = camelsfr.read_constant_cols(    # todo: the test failed
    gage_ids[:5], ["sit_area_hydro", "sol_sand", "hgl_thm_bedrock"]
)
print(attrs)
forcings = camelsfr.read_relevant_cols(
    gage_ids[:5],
	["1970-01-01", "2021-12-31"],
	var_lst = ["tsd_prec","tsd_prec_solid_frac","tsd_temp","tsd_pet_ou","tsd_pet_pe","tsd_pet_pm","tsd_wind",
               "tsd_humid","tsd_rad_dli","tsd_rad_ssi","tsd_swi_gr","tsd_swi_isba","tsd_swe_isba","tsd_temp_min",
               "tsd_temp_max"]
)
print(forcings)
streamflow = camelsfr.read_target_cols(
    gage_ids[:5],
    ["1970-01-01", "2021-12-31"],
	target_cols = ["tsd_q_l", "tsd_q_mm"],
)
print(streamflow)
attrs_types = camelsfr.get_constant_cols()
print(attrs_types)
forcing_types = camelsfr.get_relevant_cols()
print(forcing_types)
streamflow_types = camelsfr.get_target_cols()
print(streamflow_types)




camelsind = CamelsInd()
gage_ids = camelsind.read_object_ids()
assert gage_ids.size == 472
attrs = camelsind.read_constant_cols(
    gage_ids[:5], ["cwc_area","p_mean","soil_conductivity_top"]
)
print(attrs)
forcings = camelsind.read_relevant_cols(
    gage_ids[:5],
	["1980-01-01", "2020-12-31"],
        var_lst=["prcp(mm/day)", "tmax(C)", "tmin(C)", "tavg(C)", "srad_lw(w/m2)", "srad_sw(w/m2)", "wind_u(m/s)",
            "wind_v(m/s)", "wind(m/s)", "rel_hum(%)", "pet(mm/day)", "pet_gleam(mm/day)", "aet_gleam(mm/day)", "evap_canopy(kg/m2/s)",
            "evap_surface(kg/m2/s)", "sm_lvl1(kg/m2)", "sm_lvl2(kg/m2)", "sm_lvl3(kg/m2)", "sm_lvl4(kg/m2)"]
)
print(forcings)
streamflow = camelsind.read_target_cols(
    gage_ids[:5],
    ["1980-01-01", "2020-12-31"],
        # ["1980,1,1", "2020,12,31"],
        target_cols=["streamflow_observed"],
)
print(streamflow)
attrs_types = camelsind.get_constant_cols()
print(attrs_types)
forcing_types = camelsind.get_relevant_cols()
print(forcing_types)
streamflow_types = camelsind.get_target_cols()
print(streamflow_types)
