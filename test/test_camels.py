import os
import unittest

import numpy as np

import definitions
from hydrobench.data.data_camels import Camels


class CamelsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.camels_aus_path = os.path.join(definitions.DATASET_DIR, "camels", "camels_aus")
        self.camels_br_path = os.path.join(definitions.DATASET_DIR, "camels", "camels_br")
        self.camels_cl_path = os.path.join(definitions.DATASET_DIR, "camels", "camels_cl")
        self.camels_gb_path = os.path.join(definitions.DATASET_DIR, "camels", "camels_gb")
        self.camels_us_path = os.path.join(definitions.DATASET_DIR, "camels", "camels_us")
        self.camels_yr_path = os.path.join(definitions.DATASET_DIR, "camels", "camels_yr")
        self.canopex_path = os.path.join(definitions.DATASET_DIR, "canopex")
        self.lamah_ce_path = os.path.join(definitions.DATASET_DIR, "lamah_ce")
        self.aus_region = "AUS"
        self.br_region = "BR"
        self.cl_region = "CL"
        self.gb_region = "GB"
        self.us_region = "US"
        self.yr_region = "YR"
        self.ca_region = "CA"
        self.lamah_ce_region = "CE"

    def test_download_camels(self):
        camels_us = Camels(self.camels_us_path, download=True)
        self.assertTrue(
            os.path.isfile(os.path.join(self.camels_us_path, "basin_set_full_res", "HCDN_nhru_final_671.shp")))

    def test_read_camels_us(self):
        camels_us = Camels(self.camels_us_path, download=False, region=self.us_region)
        gage_ids = camels_us.read_object_ids()
        self.assertEqual(gage_ids.size, 671)
        attrs = camels_us.read_constant_cols(gage_ids[:5], var_lst=["soil_conductivity", "elev_mean", "geol_1st_class"])
        np.testing.assert_almost_equal(attrs, np.array(
            [[1.10652248, 250.31, 10.], [2.37500506, 92.68, 0.], [1.28980735, 143.8, 10.],
             [1.37329168, 247.8, 10.], [2.61515428, 310.38, 7.]]))
        forcings = camels_us.read_relevant_cols(gage_ids[:5], ["1990-01-01", "2010-01-01"],
                                                var_lst=["dayl", "prcp", "srad"])
        np.testing.assert_array_equal(forcings.shape, np.array([5, 7305, 3]))
        flows = camels_us.read_target_cols(gage_ids[:5], ["1990-01-01", "2010-01-01"], target_cols=["usgsFlow"])
        np.testing.assert_array_equal(flows.shape, np.array([5, 7305, 1]))
        streamflow_types = camels_us.get_target_cols()
        np.testing.assert_array_equal(streamflow_types, np.array(["usgsFlow"]))
        focing_types = camels_us.get_relevant_cols()
        np.testing.assert_array_equal(focing_types, np.array(['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']))
        attr_types = camels_us.get_constant_cols()
        np.testing.assert_array_equal(attr_types[:3], np.array(['gauge_lat', 'gauge_lon', 'elev_mean']))

    def test_download_camels_aus(self):
        camels_aus = Camels(self.camels_aus_path, download=True, region="AUS")
        self.assertTrue(os.path.isfile(os.path.join(self.camels_aus_path, "05_hydrometeorology", "05_hydrometeorology",
                                                    "01_precipitation_timeseries", "precipitation_AWAP.csv")))

    def test_read_camels_aus(self):
        camels_aus = Camels(self.camels_aus_path, download=False, region=self.aus_region)
        gage_ids = camels_aus.read_object_ids()
        self.assertEqual(gage_ids.size, 222)
        attrs = camels_aus.read_constant_cols(gage_ids[:5], var_lst=["catchment_area", "slope_fdc", "geol_sec"])
        np.testing.assert_array_equal(attrs, np.array(
            [[1.25773e+04, 3.66793e-01, 6.00000e+00], [1.13929e+04, 3.29998e-01, 6.00000e+00],
             [5.65300e+02, 1.78540e-02, 6.00000e+00], [4.58300e+02, 5.00234e-01, 6.00000e+00],
             [7.73170e+03, 3.74751e-01, 1.00000e+00]]))
        forcings = camels_aus.read_relevant_cols(gage_ids[:5], ["1990-01-01", "2010-01-01"],
                                                 var_lst=["precipitation_AWAP", "et_morton_actual_SILO", "tmin_SILO"])
        np.testing.assert_array_equal(forcings.shape, np.array([5, 7305, 3]))
        flows = camels_aus.read_target_cols(gage_ids[:5], ["1990-01-01", "2010-01-01"],
                                            target_cols=["streamflow_MLd", "streamflow_mmd"])
        np.testing.assert_array_equal(flows.shape, np.array([5, 7305, 2]))
        streamflow_types = camels_aus.get_target_cols()
        np.testing.assert_array_equal(streamflow_types, np.array(
            ["streamflow_MLd", "streamflow_MLd_inclInfilled", "streamflow_mmd", "streamflow_QualityCodes"]))
        focing_types = camels_aus.get_relevant_cols()
        np.testing.assert_array_equal(focing_types, np.array(
            ['precipitation_AWAP', 'precipitation_SILO', 'precipitation_var_AWAP', 'et_morton_actual_SILO',
             'et_morton_point_SILO', 'et_morton_wet_SILO', 'et_short_crop_SILO', 'et_tall_crop_SILO',
             'evap_morton_lake_SILO', 'evap_pan_SILO', 'evap_syn_SILO', 'solarrad_AWAP', 'tmax_AWAP', 'tmin_AWAP',
             'vprp_AWAP', 'mslp_SILO', 'radiation_SILO', 'rh_tmax_SILO', 'rh_tmin_SILO', 'tmax_SILO', 'tmin_SILO',
             'vp_deficit_SILO', 'vp_SILO']))
        attr_types = camels_aus.get_constant_cols()
        np.testing.assert_array_equal(attr_types[:3],
                                      np.array(['station_name', 'drainage_division', 'river_region']))

    def test_download_camels_br(self):
        camels_br = Camels(self.camels_br_path, download=True, region="BR")
        self.assertTrue(
            os.path.isfile(os.path.join(self.camels_br_path, "01_CAMELS_BR_attributes", "01_CAMELS_BR_attributes",
                                        "CAMELS_BR_attributes_description.xlsx")))

    def test_read_camels_br(self):
        camels_br = Camels(self.camels_br_path, download=False, region=self.br_region)
        gage_ids = camels_br.read_object_ids()
        self.assertEqual(gage_ids.size, 897)
        attrs = camels_br.read_constant_cols(gage_ids[:5], var_lst=["geol_class_1st", "p_mean", "runoff_ratio"])
        np.testing.assert_array_equal(attrs, np.array(
            [[8., 6.51179, 0.55336], [6., 5.38941, 0.72594], [6., 5.70191, 0.759], [6., 5.19877, 0.39463],
             [8., 5.49805, 0.38579]]))
        forcings = camels_br.read_relevant_cols(gage_ids[:5], ["1995-01-01", "2015-01-01"],
                                                var_lst=["precipitation_chirps", "evapotransp_gleam",
                                                         "potential_evapotransp_gleam", "temperature_min_cpc"])
        np.testing.assert_array_equal(forcings.shape, np.array([5, 7305, 4]))
        # start from 1995/01/01 end with 2017/04/30
        flows = camels_br.read_target_cols(gage_ids[:5], ["1995-01-01", "2015-01-01"],
                                           target_cols=["streamflow_m3s", "streamflow_mm_selected_catchments"])
        np.testing.assert_array_equal(flows.shape, np.array([5, 7305, 2]))
        streamflow_types = camels_br.get_target_cols()
        np.testing.assert_array_equal(streamflow_types, np.array(
            ['streamflow_m3s', 'streamflow_mm_selected_catchments', 'streamflow_simulated']))
        focing_types = camels_br.get_relevant_cols()
        np.testing.assert_array_equal(focing_types, np.array(
            ['precipitation_chirps', 'precipitation_mswep', 'precipitation_cpc', 'evapotransp_gleam', 'evapotransp_mgb',
             'potential_evapotransp_gleam', 'temperature_min_cpc', 'temperature_mean_cpc', 'temperature_max_cpc']))
        attr_types = camels_br.get_constant_cols()
        np.testing.assert_array_equal(attr_types[:3], np.array(['p_mean', 'pet_mean', 'et_mean']))

    def test_download_camels_cl(self):
        camels_cl = Camels(self.camels_cl_path, download=True, region="CL")
        self.assertTrue(
            os.path.isfile(os.path.join(self.camels_cl_path, "1_CAMELScl_attributes", "1_CAMELScl_attributes.txt")))

    def test_read_camels_cl(self):
        camels_cl = Camels(self.camels_cl_path, download=False, region=self.cl_region)
        gage_ids = camels_cl.read_object_ids()
        self.assertEqual(gage_ids.size, 516)
        # TODO: NaN value
        attrs = camels_cl.read_constant_cols(gage_ids[:5], var_lst=["geol_class_1st", "crop_frac"])
        np.testing.assert_array_equal(attrs, np.array([[9., 0.], [9., 53.], [9., 64.], [10., 144.], [10., 104.]]))
        # TODO: NaN value, mainly for pet_8d_modis
        forcings = camels_cl.read_relevant_cols(gage_ids[:5], ["1995-01-01", "2015-01-01"],
                                                var_lst=["pet_8d_modis", "precip_cr2met", "swe"])
        np.testing.assert_array_equal(forcings.shape, np.array([5, 7305, 3]))
        flows = camels_cl.read_target_cols(gage_ids[:5], ["1995-01-01", "2015-01-01"],
                                           target_cols=["streamflow_m3s", "streamflow_mm"])
        np.testing.assert_array_equal(flows.shape, np.array([5, 7305, 2]))
        streamflow_types = camels_cl.get_target_cols()
        np.testing.assert_array_equal(streamflow_types, np.array(['streamflow_m3s', 'streamflow_mm']))
        focing_types = camels_cl.get_relevant_cols()
        np.testing.assert_array_equal(focing_types, np.array(
            ['precip_cr2met', 'precip_chirps', 'precip_mswep', 'precip_tmpa', 'tmin_cr2met', 'tmax_cr2met',
             'tmean_cr2met', 'pet_8d_modis', 'pet_hargreaves', 'swe']))
        attr_types = camels_cl.get_constant_cols()
        np.testing.assert_array_equal(attr_types[:3], np.array(['gauge_name', 'gauge_lat', 'gauge_lon']))

    def test_download_camels_gb(self):
        camels_gb = Camels(self.camels_gb_path, download=True, region=self.gb_region)
        self.assertTrue(
            os.path.isfile(os.path.join(self.camels_gb_path, "8344e4f3-d2ea-44f5-8afa-86d2987543a9",
                                        "8344e4f3-d2ea-44f5-8afa-86d2987543a9", "data",
                                        "CAMELS_GB_climatic_attributes.csv")))

    def test_read_camels_gb(self):
        camels_gb = Camels(self.camels_gb_path, download=False, region=self.gb_region)
        gage_ids = camels_gb.read_object_ids()
        self.assertEqual(gage_ids.size, 671)
        attrs = camels_gb.read_constant_cols(gage_ids[:5], var_lst=["p_mean", "slope_fdc", "gauge_name"])
        np.testing.assert_array_equal(attrs, np.array(
            [[2.29, 1.94, 596.], [2.31, 1.95, 670.], [2.65, 4.01, 647.], [2.31, 1.54, 393.], [2.29, 1.47, 217.]]))
        forcings = camels_gb.read_relevant_cols(gage_ids[:5], ["1995-01-01", "2015-01-01"],
                                                var_lst=["precipitation", "pet", "temperature", "peti"])
        np.testing.assert_array_equal(forcings.shape, np.array([5, 7305, 4]))
        flows = camels_gb.read_target_cols(gage_ids[:5], ["1995-01-01", "2015-01-01"],
                                           target_cols=["discharge_spec", "discharge_vol"])
        np.testing.assert_array_equal(flows.shape, np.array([5, 7305, 2]))
        streamflow_types = camels_gb.get_target_cols()
        np.testing.assert_array_equal(streamflow_types, np.array(["discharge_spec", "discharge_vol"]))
        focing_types = camels_gb.get_relevant_cols()
        np.testing.assert_array_equal(focing_types, np.array(
            ["precipitation", "pet", "temperature", "peti", "humidity", "shortwave_rad", "longwave_rad", "windspeed"]))
        attr_types = camels_gb.get_constant_cols()
        np.testing.assert_array_equal(attr_types[:3], np.array(["p_mean", "pet_mean", "aridity"]))

    def test_download_camels_yr(self):
        camels_yr = Camels(self.camels_yr_path, download=True, region=self.yr_region)
        self.assertTrue(
            os.path.isfile(os.path.join(self.camels_yr_path, "8344e4f3-d2ea-44f5-8afa-86d2987543a9",
                                        "8344e4f3-d2ea-44f5-8afa-86d2987543a9", "data",
                                        "CAMELS_GB_climatic_attributes.csv")))

    def test_read_camels_yr(self):
        camels_yr = Camels(self.camels_yr_path, download=False, region=self.yr_region)
        gage_ids = camels_yr.read_object_ids()
        self.assertEqual(gage_ids.size, 102)
        attrs = camels_yr.read_constant_cols(gage_ids[:5], var_lst=["area", "barren", "bdticm"])
        np.testing.assert_almost_equal(attrs, np.array(
            [[3.11520000e+04, 2.98706264e-03, 3.33904449e+03], [4.27056000e+05, 2.30162622e-02, 1.80570119e+03],
             [3.80128000e+05, 2.47549979e-02, 1.77874628e+03], [7.33561000e+05, 1.41340180e-02, 2.01143843e+03],
             [2.04213000e+05, 7.75394506e-03, 1.53321208e+03]]), decimal=3)
        forcings = camels_yr.read_relevant_cols(gage_ids[:5], ["1995-01-01", "2015-01-01"],
                                                var_lst=["pre", "evp", "gst_mean", "prs_mean"])
        np.testing.assert_array_equal(forcings.shape, np.array([5, 7305, 4]))
        flows = camels_yr.read_target_cols(gage_ids[:5], ["1995-01-01", "2015-01-01"], target_cols=["normalized_q"])
        np.testing.assert_array_equal(flows.shape, np.array([5, 7305, 1]))
        streamflow_types = camels_yr.get_target_cols()
        np.testing.assert_array_equal(streamflow_types, np.array(["normalized_q"]))
        focing_types = camels_yr.get_relevant_cols()
        np.testing.assert_array_equal(focing_types, np.array(
            ["pre", "evp", "gst_mean", "prs_mean", "tem_mean", "rhu", "win_mean", "gst_min", "prs_min", "tem_min",
             "gst_max", "prs_max", "tem_max", "ssd", "win_max"]))
        attr_types = camels_yr.get_constant_cols()
        np.testing.assert_array_equal(attr_types[:3], np.array(["area", "barren", "bdticm"]))

    def test_download_canopex(self):
        canopex = Camels(self.canopex_path, download=True, region=self.ca_region)
        self.assertTrue(
            os.path.isfile(os.path.join(self.canopex_path, "CANOPEX_NRCAN_ASCII", "CANOPEX_NRCAN_ASCII", "1.dly")))

    def test_read_canopex_data(self):
        canopex = Camels(self.canopex_path, download=False, region=self.ca_region)
        gage_ids = canopex.read_object_ids()
        self.assertEqual(gage_ids.size, 611)
        attrs = canopex.read_constant_cols(gage_ids[:5],
                                           var_lst=["Drainage_Area_km2", "Land_Use_Grass_frac", "Permeability_logk_m2"])
        np.testing.assert_almost_equal(attrs, np.array(
            [[3.28438700e+02, 6.94000000e-02, - 1.35251586e+01], [3.43515600e+02, 6.16000000e-02, - 1.42367348e+01],
             [1.45554950e+03, 3.59000000e-02, - 1.50071080e+01], [5.64820300e+02, 3.05000000e-02, - 1.51002546e+01],
             [1.05383090e+03, 3.27000000e-02, - 1.51999998e+01]]), decimal=3)
        forcings = canopex.read_relevant_cols(gage_ids[:5], ["1990-01-01", "2010-01-01"],
                                              var_lst=["prcp", "tmax", "tmin"])
        np.testing.assert_array_equal(forcings.shape, np.array([5, 7305, 3]))
        flows = canopex.read_target_cols(gage_ids[:5], ["1990-01-01", "2010-01-01"], target_cols=["discharge"])
        np.testing.assert_array_equal(flows.shape, np.array([5, 7305, 1]))
        streamflow_types = canopex.get_target_cols()
        np.testing.assert_array_equal(streamflow_types, np.array(["discharge"]))
        focing_types = canopex.get_relevant_cols()
        np.testing.assert_array_equal(focing_types, np.array(["prcp", "tmax", "tmin"]))
        attr_types = canopex.get_constant_cols()
        np.testing.assert_array_equal(attr_types[:3], np.array(["Source", "Name", "Official_ID"]))

    def test_download_lamah_ce(self):
        lamah_ce = Camels(self.lamah_ce_path, download=True, region=self.lamah_ce_region)
        self.assertTrue(
            os.path.isfile(os.path.join(self.lamah_ce_path, "CANOPEX_NRCAN_ASCII", "CANOPEX_NRCAN_ASCII", "1.dly")))

    def test_read_lamah_ce(self):
        lamah_ce = Camels(self.lamah_ce_path, download=False, region=self.lamah_ce_region)
        gage_ids = lamah_ce.read_object_ids()
        self.assertEqual(gage_ids.size, 859)
        attrs = lamah_ce.read_constant_cols(gage_ids[:5], var_lst=['area_calc', 'elev_mean', 'elev_med'])
        np.testing.assert_almost_equal(attrs, np.array(
            [[4668.379, 1875., 1963.], [102.287, 1775., 1827.], [536.299, 1844., 1916.], [66.286, 1894., 1907.],
             [72.448, 1774., 1796.]]), decimal=3)
        forcings = lamah_ce.read_relevant_cols(gage_ids[:5], ["1990-01-01", "2010-01-01"],
                                               var_lst=["2m_temp_max", "prec", "volsw_4"])
        np.testing.assert_array_equal(forcings.shape, np.array([5, 7305, 3]))
        flows = lamah_ce.read_target_cols(gage_ids[:5], ["1990-01-01", "2010-01-01"], target_cols=["qobs"])
        np.testing.assert_array_equal(flows.shape, np.array([5, 7305, 1]))
        streamflow_types = lamah_ce.get_target_cols()
        np.testing.assert_array_equal(streamflow_types, np.array(["qobs"]))
        focing_types = lamah_ce.get_relevant_cols()
        np.testing.assert_array_equal(focing_types, np.array(
            ["2m_temp_max", "2m_temp_mean", "2m_temp_min", "2m_dp_temp_max", "2m_dp_temp_mean",
             "2m_dp_temp_min", "10m_wind_u", "10m_wind_v", "fcst_alb", "lai_high_veg", "lai_low_veg",
             "swe", "surf_net_solar_rad_max", "surf_net_solar_rad_mean", "surf_net_therm_rad_max",
             "surf_net_therm_rad_mean", "surf_press", "total_et", "prec", "volsw_123", "volsw_4"]))
        attr_types = lamah_ce.get_constant_cols()
        np.testing.assert_array_equal(attr_types, np.array(
            ['area_calc', 'elev_mean', 'elev_med', 'elev_std', 'elev_ran', 'slope_mean', 'mvert_dist', 'mvert_ang',
             'elon_ratio', 'strm_dens', 'p_mean', 'et0_mean', 'eta_mean', 'arid_1', 'arid_2', 'p_season', 'frac_snow',
             'hi_prec_fr', 'hi_prec_du', 'hi_prec_ti', 'lo_prec_fr', 'lo_prec_du', 'lo_prec_ti', 'lc_dom', 'agr_fra',
             'bare_fra', 'forest_fra', 'glac_fra', 'lake_fra', 'urban_fra', 'lai_max', 'lai_diff', 'ndvi_max',
             'ndvi_min', 'gvf_max', 'gvf_diff', 'bedrk_dep', 'root_dep', 'soil_poros', 'soil_condu', 'soil_tawc',
             'sand_fra', 'silt_fra', 'clay_fra', 'grav_fra', 'oc_fra', 'gc_dom', 'gc_ig_fra', 'gc_mt_fra', 'gc_pa_fra',
             'gc_pb_fra', 'gc_pi_fra', 'gc_py_fra', 'gc_sc_fra', 'gc_sm_fra', 'gc_ss_fra', 'gc_su_fra', 'gc_va_fra',
             'gc_vb_fra', 'gc_wb_fra', 'geol_perme', 'geol_poros']))

    def test_ca_p_mean(self):
        canopex = Camels(self.canopex_path, download=False, region=self.ca_region)
        gage_ids = canopex.read_object_ids()
        p_mean = canopex.read_mean_prep(gage_ids[:5])
        np.testing.assert_almost_equal(p_mean, np.array([2.91712073, 3.14145935, 3.12958083, 3.09248435, 3.04431583]))


if __name__ == '__main__':
    unittest.main()
