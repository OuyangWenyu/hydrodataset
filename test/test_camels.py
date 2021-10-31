import os
import unittest

import numpy as np

import definitions
from hydrobench.data.data_camels import Camels


class CamelsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.camels_path = os.path.join(definitions.DATASET_DIR, "camels", "camels_us")
        self.camels_aus_path = os.path.join(definitions.DATASET_DIR, "camels", "camels_aus")
        self.camels_br_path = os.path.join(definitions.DATASET_DIR, "camels", "camels_br")
        self.camels_cl_path = os.path.join(definitions.DATASET_DIR, "camels", "camels_cl")
        self.aus_region = "AUS"
        self.br_region = "BR"
        self.cl_region = "CL"

    def test_download_camels(self):
        camels = Camels(self.camels_path, download=True)
        self.assertTrue(os.path.isfile(os.path.join(self.camels_path, "basin_set_full_res", "HCDN_nhru_final_671.shp")))

    def test_read_camels_us(self):
        camels = Camels(self.camels_path, download=True)
        flows = camels.read_target_cols(camels.read_object_ids()[:5], t_range=["1990-01-01", "2010-01-01"],
                                        target_cols=["usgsFlow"])
        np.testing.assert_array_equal(flows.shape, np.array([5, 7305, 1]))

    def test_download_camels_aus(self):
        camels_aus = Camels(self.camels_aus_path, download=True, region="AUS")
        self.assertTrue(os.path.isfile(os.path.join(self.camels_aus_path, "05_hydrometeorology", "05_hydrometeorology",
                                                    "01_precipitation_timeseries", "precipitation_AWAP.csv")))

    def test_read_camels_aus_data(self):
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

    def test_read_camels_br_data(self):
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

    def test_read_camels_cl_data(self):
        camels_cl = Camels(self.camels_cl_path, download=False, region=self.cl_region)
        gage_ids = camels_cl.read_object_ids()
        self.assertEqual(gage_ids.size, 516)
        # TODO: NaN value
        attrs = camels_cl.read_constant_cols(gage_ids[:5], var_lst=["geol_class_1st", "crop_frac"])
        np.testing.assert_array_equal(attrs, np.array([[9., 0.], [9., 53.], [9., 64.], [10., 144.], [10., 104.]]))
        # TODO: pet_8d_modis and NaN value
        forcings = camels_cl.read_relevant_cols(gage_ids[:5], ["1995-01-01", "2015-01-01"],
                                                var_lst=["precip_cr2met", "swe"])
        np.testing.assert_array_equal(forcings.shape, np.array([5, 7305, 2]))
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


if __name__ == '__main__':
    unittest.main()
