from hydrodataset import CamelsFr

def test_read_forcing():
    camelsfr = CamelsFr()
    gage_ids = camelsfr.read_object_ids()
    print(gage_ids)
    forcings = camelsfr.read_relevant_cols(
        gage_ids[:5],
        ["1970-01-01", "2022-01-01"],
		var_lst = ["tsd_prec","tsd_prec_solid_frac","tsd_temp","tsd_pet_ou","tsd_pet_pe","tsd_pet_pm","tsd_wind",
                   "tsd_humid","tsd_rad_dli","tsd_rad_ssi","tsd_swi_gr","tsd_swi_isba","tsd_swe_isba","tsd_temp_min",
                   "tsd_temp_max"]
    )
    print(forcings)

def test_read_streamflow():
    camelsfr = CamelsFr()
    gage_ids = camelsfr.read_object_ids()
    streamflow = camelsfr.read_target_cols(
        gage_ids[:5],
        ["1970-01-01", "2022-01-01"],
		target_cols = ["tsd_q_l", "tsd_q_mm"],
    )
    print(streamflow)

def test_read_attr():
    camelsfr = CamelsFr()
    gage_ids = camelsfr.read_object_ids()
    attributes = camelsfr.read_constant_cols(
        gage_ids[:5],
        ["geo_dom_class", "hgl_krs_karstic", "hgl_thm_bedrock"]
    )
    print(attributes)

def test_cache_forcing():
    camelsfr = CamelsFr()
    cacheforcing = camelsfr.cache_forcing_xrdataset()

def test_cache_streamflow():
    camelsfr = CamelsFr()
    cachestreamflow = camelsfr.cache_streamflow_xrdataset()

def test_cache_attributes():
    camelsfr = CamelsFr()
    cacheatributes = camelsfr.cache_attributes_xrdataset()

def test_cache_nestedness_df():
    camelsfr = CamelsFr()
    cachenestedness = camelsfr.cache_nestedness_df()
    print(cachenestedness)

def test_cache_xrdataset():
    camelsfr = CamelsFr()
    cachexrdataset = camelsfr.cache_xrdataset()
    print(cachexrdataset)

def test_read_area_meanprcp():
    camelsfr = CamelsFr()
    gage_ids = camelsfr.read_object_ids()
    areas = camelsfr.read_area(gage_ids[:5])
    mean_prcp = camelsfr.read_mean_prcp(gage_ids[:5])
    print(areas.values)
    print(mean_prcp.values)

def test_read_nestedness_csv():
    camelsfr = CamelsFr()
    nestedness = camelsfr.read_nestedness_csv()
    print(nestedness)

def test_set_data_source_describe():
    camelsfr = CamelsFr()
    describle = camelsfr.set_data_source_describe()
    print("\n")
    print(describle)
# OrderedDict([('CAMELS_DIR', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_fr')),
# ('CAMELS_FLOW_DIR', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_fr/CAMELS_FR_time_series/daily')),
# ('CAMELS_FORCING_DIR', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_fr/CAMELS_FR_time_series/daily')),
# ('CAMELS_ATTR_DIR', [WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_fr/CAMELS_FR_attributes/static_attributes'),
# WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_fr/CAMELS_FR_attributes/time_series_statistics')]),
# ('CAMELS_ATTR_KEY_LST', ['geology', 'human_influences_dams', 'hydrogeology', 'land_cover', 'station_general', 'topography_general',
# 'climatic_statistics', 'hydroclimatic_statistics_joint_availability_yearly', 'hydrological_signatures', 'hydrometry_statistics']),
# ('CAMELS_GAUGE_FILE', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_fr/CAMELS_FR_attributes/static_attributes/CAMELS_FR_geology_attributes.csv')),
# ('CAMELS_NESTEDNESS_FILE', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_fr/CAMELS_FR_geography/CAMELS_FR_catchment_nestedness_information.csv')),
# ('CAMELS_BASINS_SHP', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_fr/CAMELS_FR_geography/CAMELS_FR_catchment_boundaries.gpkg'))])
