import os
import pytest

import numpy as np

import definitions
from hydrodataset.data.data_camels import Camels


@pytest.fixture
def camels_aus_path():
    return os.path.join(definitions.DATASET_DIR, "camels", "camels_aus")


@pytest.fixture
def camels_br_path():
    return os.path.join(definitions.DATASET_DIR, "camels", "camels_br")


@pytest.fixture
def camels_cl_path():
    return os.path.join(definitions.DATASET_DIR, "camels", "camels_cl")


@pytest.fixture
def camels_gb_path():
    return os.path.join(definitions.DATASET_DIR, "camels", "camels_gb")


@pytest.fixture
def camels_us_path():
    return os.path.join(definitions.DATASET_DIR, "camels", "camels_us")


@pytest.fixture
def camels_yr_path():
    return os.path.join(definitions.DATASET_DIR, "camels", "camels_yr")


@pytest.fixture
def canopex_path():
    return os.path.join(definitions.DATASET_DIR, "canopex")


@pytest.fixture
def lamah_ce_path():
    return os.path.join(definitions.DATASET_DIR, "lamah_ce")


@pytest.fixture
def aus_region():
    return "AUS"


@pytest.fixture
def br_region():
    return "BR"


@pytest.fixture
def cl_region():
    return "CL"


@pytest.fixture
def gb_region():
    return "GB"


@pytest.fixture
def us_region():
    return "US"


@pytest.fixture
def yr_region():
    return "YR"


@pytest.fixture
def ca_region():
    return "CA"


@pytest.fixture
def lamah_ce_region():
    return "CE"


def test_download_camels(camels_us_path):
    camels_us = Camels(camels_us_path, download=True)
    assert os.path.isfile(
        os.path.join(camels_us_path, "basin_set_full_res", "HCDN_nhru_final_671.shp")
    )
    assert os.path.isdir(
        os.path.join(camels_us_path, "camels_streamflow", "camels_streamflow")
    )


def test_read_camels_streamflow(camels_us_path, us_region):
    camels_us = Camels(camels_us_path, download=False, region=us_region)
    gage_ids = camels_us.read_object_ids()
    flows1 = camels_us.read_target_cols(
        gage_ids[:5], ["2013-01-01", "2018-01-01"], target_cols=["usgsFlow"]
    )
    print(flows1)
    flows2 = camels_us.read_target_cols(
        gage_ids[:5], ["2015-01-01", "2018-01-01"], target_cols=["usgsFlow"]
    )
    print(flows2)


def test_read_camels_us(camels_us_path, us_region):
    camels_us = Camels(camels_us_path, download=False, region=us_region)
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


def test_download_camels_aus(camels_aus_path):
    camels_aus = Camels(camels_aus_path, download=True, region="AUS")
    assert os.path.isfile(
        os.path.join(
            camels_aus_path,
            "05_hydrometeorology",
            "05_hydrometeorology",
            "01_precipitation_timeseries",
            "precipitation_AWAP.csv",
        )
    )


def test_read_camels_aus(camels_aus_path, aus_region):
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


def test_download_camels_br(camels_br_path):
    camels_br = Camels(camels_br_path, download=True, region="BR")
    assert os.path.isfile(
        os.path.join(
            camels_br_path,
            "01_CAMELS_BR_attributes",
            "01_CAMELS_BR_attributes",
            "CAMELS_BR_attributes_description.xlsx",
        )
    )


def test_read_camels_br(camels_br_path, br_region):
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


def test_download_camels_cl(camels_cl_path):
    camels_cl = Camels(camels_cl_path, download=True, region="CL")
    assert os.path.isfile(
        os.path.join(
            camels_cl_path, "1_CAMELScl_attributes", "1_CAMELScl_attributes.txt"
        )
    )


def test_read_camels_cl(camels_cl_path, cl_region):
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


def test_download_camels_gb(camels_gb_path, gb_region):
    camels_gb = Camels(camels_gb_path, download=True, region=gb_region)
    assert os.path.isfile(
        os.path.join(
            camels_gb_path,
            "8344e4f3-d2ea-44f5-8afa-86d2987543a9",
            "8344e4f3-d2ea-44f5-8afa-86d2987543a9",
            "data",
            "CAMELS_GB_climatic_attributes.csv",
        )
    )


def test_read_camels_gb(camels_gb_path, gb_region):
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


def test_download_camels_yr(camels_yr_path, yr_region):
    camels_yr = Camels(camels_yr_path, download=True, region=yr_region)
    assert os.path.isfile(
        os.path.join(
            camels_yr_path,
            "9_Normal_Camels_YR",
            "1_Normal_Camels_YR_basin_data",
            "0146",
            "attributes.json",
        )
    )


def test_read_camels_yr(camels_yr_path, yr_region):
    camels_yr = Camels(camels_yr_path, download=False, region=yr_region)
    gage_ids = camels_yr.read_object_ids()
    assert gage_ids.size == 102
    attrs = camels_yr.read_constant_cols(
        gage_ids[:5], var_lst=["area", "barren", "bdticm"]
    )
    np.testing.assert_almost_equal(
        attrs,
        np.array(
            [
                [3.11520000e04, 2.98706264e-03, 3.33904449e03],
                [4.27056000e05, 2.30162622e-02, 1.80570119e03],
                [3.80128000e05, 2.47549979e-02, 1.77874628e03],
                [7.33561000e05, 1.41340180e-02, 2.01143843e03],
                [2.04213000e05, 7.75394506e-03, 1.53321208e03],
            ]
        ),
        decimal=3,
    )
    forcings = camels_yr.read_relevant_cols(
        gage_ids[:5],
        ["1995-01-01", "2015-01-01"],
        var_lst=["pre", "evp", "gst_mean", "prs_mean"],
    )
    np.testing.assert_array_equal(forcings.shape, np.array([5, 7305, 4]))
    flows = camels_yr.read_target_cols(
        gage_ids[:5], ["1995-01-01", "2015-01-01"], target_cols=["normalized_q"]
    )
    np.testing.assert_array_equal(flows.shape, np.array([5, 7305, 1]))
    streamflow_types = camels_yr.get_target_cols()
    np.testing.assert_array_equal(streamflow_types, np.array(["normalized_q"]))
    focing_types = camels_yr.get_relevant_cols()
    np.testing.assert_array_equal(
        focing_types,
        np.array(
            [
                "pre",
                "evp",
                "gst_mean",
                "prs_mean",
                "tem_mean",
                "rhu",
                "win_mean",
                "gst_min",
                "prs_min",
                "tem_min",
                "gst_max",
                "prs_max",
                "tem_max",
                "ssd",
                "win_max",
            ]
        ),
    )
    attr_types = camels_yr.get_constant_cols()
    np.testing.assert_array_equal(
        attr_types[:3], np.array(["area", "barren", "bdticm"])
    )


def test_download_canopex(canopex_path, ca_region):
    canopex = Camels(canopex_path, download=True, region=ca_region)
    assert os.path.isfile(
        os.path.join(
            canopex_path, "CANOPEX_NRCAN_ASCII", "CANOPEX_NRCAN_ASCII", "1.dly"
        )
    )


def test_read_canopex_data(canopex_path, ca_region):
    canopex = Camels(canopex_path, download=False, region=ca_region)
    gage_ids = canopex.read_object_ids()
    assert gage_ids.size == 611
    attrs = canopex.read_constant_cols(
        gage_ids[:5],
        var_lst=["Drainage_Area_km2", "Land_Use_Grass_frac", "Permeability_logk_m2"],
    )
    np.testing.assert_almost_equal(
        attrs,
        np.array(
            [
                [3.28438700e02, 6.94000000e-02, -1.35251586e01],
                [3.43515600e02, 6.16000000e-02, -1.42367348e01],
                [1.45554950e03, 3.59000000e-02, -1.50071080e01],
                [5.64820300e02, 3.05000000e-02, -1.51002546e01],
                [1.05383090e03, 3.27000000e-02, -1.51999998e01],
            ]
        ),
        decimal=3,
    )
    forcings = canopex.read_relevant_cols(
        gage_ids[:5], ["1990-01-01", "2010-01-01"], var_lst=["prcp", "tmax", "tmin"]
    )
    np.testing.assert_array_equal(forcings.shape, np.array([5, 7305, 3]))
    flows = canopex.read_target_cols(
        gage_ids[:5], ["1990-01-01", "2010-01-01"], target_cols=["discharge"]
    )
    np.testing.assert_array_equal(flows.shape, np.array([5, 7305, 1]))
    streamflow_types = canopex.get_target_cols()
    np.testing.assert_array_equal(streamflow_types, np.array(["discharge"]))
    focing_types = canopex.get_relevant_cols()
    np.testing.assert_array_equal(focing_types, np.array(["prcp", "tmax", "tmin"]))
    attr_types = canopex.get_constant_cols()
    np.testing.assert_array_equal(
        attr_types[:3], np.array(["Source", "Name", "Official_ID"])
    )


def test_download_lamah_ce(lamah_ce_path, lamah_ce_region):
    lamah_ce = Camels(lamah_ce_path, download=True, region=lamah_ce_region)
    assert os.path.isfile(
        os.path.join(
            lamah_ce_path,
            "2_LamaH-CE_daily",
            "A_basins_total_upstrm",
            "2_timeseries",
            "daily",
            "ID_882.csv",
        )
    )


def test_read_lamah_ce(lamah_ce_path, lamah_ce_region):
    lamah_ce = Camels(lamah_ce_path, download=False, region=lamah_ce_region)
    gage_ids = lamah_ce.read_object_ids()
    assert gage_ids.size == 859
    attrs = lamah_ce.read_constant_cols(
        gage_ids[:5], var_lst=["area_calc", "elev_mean", "elev_med"]
    )
    np.testing.assert_almost_equal(
        attrs,
        np.array(
            [
                [4668.379, 1875.0, 1963.0],
                [102.287, 1775.0, 1827.0],
                [536.299, 1844.0, 1916.0],
                [66.286, 1894.0, 1907.0],
                [72.448, 1774.0, 1796.0],
            ]
        ),
        decimal=3,
    )
    forcings = lamah_ce.read_relevant_cols(
        gage_ids[:5],
        ["1990-01-01", "2010-01-01"],
        var_lst=["2m_temp_max", "prec", "volsw_4"],
    )
    np.testing.assert_array_equal(forcings.shape, np.array([5, 7305, 3]))
    flows = lamah_ce.read_target_cols(
        gage_ids[:5], ["1990-01-01", "2010-01-01"], target_cols=["qobs"]
    )
    np.testing.assert_array_equal(flows.shape, np.array([5, 7305, 1]))
    streamflow_types = lamah_ce.get_target_cols()
    np.testing.assert_array_equal(streamflow_types, np.array(["qobs"]))
    focing_types = lamah_ce.get_relevant_cols()
    np.testing.assert_array_equal(
        focing_types,
        np.array(
            [
                "2m_temp_max",
                "2m_temp_mean",
                "2m_temp_min",
                "2m_dp_temp_max",
                "2m_dp_temp_mean",
                "2m_dp_temp_min",
                "10m_wind_u",
                "10m_wind_v",
                "fcst_alb",
                "lai_high_veg",
                "lai_low_veg",
                "swe",
                "surf_net_solar_rad_max",
                "surf_net_solar_rad_mean",
                "surf_net_therm_rad_max",
                "surf_net_therm_rad_mean",
                "surf_press",
                "total_et",
                "prec",
                "volsw_123",
                "volsw_4",
            ]
        ),
    )
    attr_types = lamah_ce.get_constant_cols()
    np.testing.assert_array_equal(
        attr_types,
        np.array(
            [
                "area_calc",
                "elev_mean",
                "elev_med",
                "elev_std",
                "elev_ran",
                "slope_mean",
                "mvert_dist",
                "mvert_ang",
                "elon_ratio",
                "strm_dens",
                "p_mean",
                "et0_mean",
                "eta_mean",
                "arid_1",
                "arid_2",
                "p_season",
                "frac_snow",
                "hi_prec_fr",
                "hi_prec_du",
                "hi_prec_ti",
                "lo_prec_fr",
                "lo_prec_du",
                "lo_prec_ti",
                "lc_dom",
                "agr_fra",
                "bare_fra",
                "forest_fra",
                "glac_fra",
                "lake_fra",
                "urban_fra",
                "lai_max",
                "lai_diff",
                "ndvi_max",
                "ndvi_min",
                "gvf_max",
                "gvf_diff",
                "bedrk_dep",
                "root_dep",
                "soil_poros",
                "soil_condu",
                "soil_tawc",
                "sand_fra",
                "silt_fra",
                "clay_fra",
                "grav_fra",
                "oc_fra",
                "gc_dom",
                "gc_ig_fra",
                "gc_mt_fra",
                "gc_pa_fra",
                "gc_pb_fra",
                "gc_pi_fra",
                "gc_py_fra",
                "gc_sc_fra",
                "gc_sm_fra",
                "gc_ss_fra",
                "gc_su_fra",
                "gc_va_fra",
                "gc_vb_fra",
                "gc_wb_fra",
                "geol_perme",
                "geol_poros",
            ]
        ),
    )


def test_ca_p_mean(canopex_path, ca_region):
    canopex = Camels(canopex_path, download=False, region=ca_region)
    gage_ids = canopex.read_object_ids()
    p_mean = canopex.read_mean_prep(gage_ids[:5])
    np.testing.assert_almost_equal(
        p_mean, np.array([2.91712073, 3.14145935, 3.12958083, 3.09248435, 3.04431583])
    )
