from hydrodataset import CamelsBr

def test_read_forcing():
    camelsbr = CamelsBr()
    gage_ids = camelsbr.read_object_ids()
    print(gage_ids)
    forcings = camelsbr.read_relevant_cols(
        gage_ids[:5],
        ["1995-01-01", "2015-01-01"],
		var_lst = ["precipitation_chirps", "precipitation_mswep", "precipitation_cpc", "evapotransp_gleam", "evapotransp_mgb",
                   "potential_evapotransp_gleam", "temperature_min_cpc", "temperature_mean_cpc", "temperature_max_cpc"]
    )
    print(forcings)

def test_read_streamflow():
    camelsbr = CamelsBr()
    gage_ids = camelsbr.read_object_ids()
    streamflow = camelsbr.read_target_cols(
        gage_ids[:5],
        ["1995-01-01", "2015-01-01"],
		target_cols = ["streamflow_mm_selected_catchments" ],
    )
    print(streamflow)

def test_read_attr():
    camelsbr = CamelsBr()
    gage_ids = camelsbr.read_object_ids()
    attributes = camelsbr.read_constant_cols(
        gage_ids[:5],
        ["p_mean", "area", "crop_perc"]
    )
    print(attributes)

def test_cache_forcing():
    camelsbr = CamelsBr()
    cacheforcing = camelsbr.cache_forcing_xrdataset()

def test_cache_streamflow():
    camelsbr = CamelsBr()
    cachestreamflow = camelsbr.cache_streamflow_xrdataset()

def test_cache_attributes():
    camelsbr = CamelsBr()
    cacheatributes = camelsbr.cache_attributes_xrdataset()

def test_cache_xrdataset():
    camelsbr = CamelsBr()
    cachexrdataset = camelsbr.cache_xrdataset()

def test_read_area_meanprcp():
    camelsbr = CamelsBr()
    gage_ids = camelsbr.read_object_ids()
    areas = camelsbr.read_area(gage_ids[:5])
    mean_prcp = camelsbr.read_mean_prcp(gage_ids[:5])
    print(areas.values)
    print(mean_prcp.values)

def test_set_data_source_describe():
    camelsbr = CamelsBr()
    describle = camelsbr.set_data_source_describe()
    print("\n")
    print(describle)
# OrderedDict([('CAMELS_DIR', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_br')),
# ('CAMELS_FLOW_DIR', [WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_br/03_CAMELS_BR_streamflow_mm_selected_catchments/03_CAMELS_BR_streamflow_mm_selected_catchments')]),
# ('CAMELS_FORCING_DIR', [WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_br/05_CAMELS_BR_precipitation_chirps/05_CAMELS_BR_precipitation_chirps'),
# WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_br/06_CAMELS_BR_precipitation_mswep/06_CAMELS_BR_precipitation_mswep'),
# WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_br/07_CAMELS_BR_precipitation_cpc/07_CAMELS_BR_precipitation_cpc'),
# WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_br/08_CAMELS_BR_evapotransp_gleam/08_CAMELS_BR_evapotransp_gleam'),
# WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_br/09_CAMELS_BR_evapotransp_mgb/09_CAMELS_BR_evapotransp_mgb'),
# WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_br/10_CAMELS_BR_potential_evapotransp_gleam/10_CAMELS_BR_potential_evapotransp_gleam'),
# WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_br/11_CAMELS_BR_temperature_min_cpc/11_CAMELS_BR_temperature_min_cpc'),
# WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_br/12_CAMELS_BR_temperature_mean_cpc/12_CAMELS_BR_temperature_mean_cpc'),
# WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_br/13_CAMELS_BR_temperature_max_cpc/13_CAMELS_BR_temperature_max_cpc')]),
# ('CAMELS_ATTR_DIR', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_br/01_CAMELS_BR_attributes/01_CAMELS_BR_attributes')),
# ('CAMELS_ATTR_KEY_LST', ['climate', 'geology', 'human_intervention', 'hydrology', 'land_cover', 'quality_check', 'soil', 'topography']),
# ('CAMELS_GAUGE_FILE', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_br/01_CAMELS_BR_attributes/01_CAMELS_BR_attributes/camels_br_topography.txt')),
# ('CAMELS_NESTEDNESS_FILE', None),
# ('CAMELS_BASINS_SHP_FILE', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_br/14_CAMELS_BR_catchment_boundaries/14_CAMELS_BR_catchment_boundaries/camels_br_catchments.shp'))])
