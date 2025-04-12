from hydrodataset import CamelsCl

def test_read_forcing():
    camelscl = CamelsCl()
    gage_ids = camelscl.read_object_ids()
    print(gage_ids)
    forcings = camelscl.read_relevant_cols(
        gage_ids[:5],
        ["1995-01-01", "2015-01-01"],
		var_lst = ["precip_cr2met", "precip_chirps", "precip_mswep", "precip_tmpa", "tmin_cr2met", "tmax_cr2met", "tmean_cr2met", "pet_8d_modis", "pet_hargreaves", "swe",]
    )
    print(forcings)

def test_read_streamflow():
    camelscl = CamelsCl()
    gage_ids = camelscl.read_object_ids()
    streamflow = camelscl.read_target_cols(
        gage_ids[:5],
        ["1995-01-01", "2015-01-01"],
		target_cols = ["streamflow_m3s", "streamflow_mm"],
    )
    print(streamflow)

def test_read_attr():
    camelscl = CamelsCl()
    gage_ids = camelscl.read_object_ids()
    attributes = camelscl.read_constant_cols(
        gage_ids[:5],
        ["p_mean_cr2met", "area", "slope_fdc"]
    )
    print(attributes)

def test_cache_forcing():
    camelscl = CamelsCl()
    cacheforcing = camelscl.cache_forcing_xrdataset()

def test_cache_streamflow():
    camelscl = CamelsCl()
    cachestreamflow = camelscl.cache_streamflow_xrdataset()

def test_cache_attributes():
    camelscl = CamelsCl()
    cacheatributes = camelscl.cache_attributes_xrdataset()

def test_cache_xrdataset():
    camelscl = CamelsCl()
    cachexrdataset = camelscl.cache_xrdataset()

def test_read_area_meanprcp():
    camelscl = CamelsCl()
    gage_ids = camelscl.read_object_ids()
    areas = camelscl.read_area(gage_ids[:5])
    mean_prcp = camelscl.read_mean_prcp(gage_ids[:5])
    print(areas.values)
    print(mean_prcp.values)

def test_set_data_source_describe():
    camelscl = CamelsCl()
    describle = camelscl.set_data_source_describe()
    print("\n")
    print(describle)
# OrderedDict([('CAMELS_DIR', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_cl')),
# ('CAMELS_FLOW_DIR', [WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_cl/2_CAMELScl_streamflow_m3s'),
# WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_cl/3_CAMELScl_streamflow_mm')]),
# ('CAMELS_FORCING_DIR', [WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_cl/4_CAMELScl_precip_cr2met'),
# WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_cl/5_CAMELScl_precip_chirps'),
# WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_cl/6_CAMELScl_precip_mswep'),
# WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_cl/7_CAMELScl_precip_tmpa'),
# WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_cl/8_CAMELScl_tmin_cr2met'),
# WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_cl/9_CAMELScl_tmax_cr2met'),
# WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_cl/10_CAMELScl_tmean_cr2met'),
# WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_cl/11_CAMELScl_pet_8d_modis'),
# WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_cl/12_CAMELScl_pet_hargreaves'),
# WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_cl/13_CAMELScl_swe')]),
# ('CAMELS_ATTR_DIR', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_cl/1_CAMELScl_attributes')),
# ('CAMELS_GAUGE_FILE', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_cl/1_CAMELScl_attributes/1_CAMELScl_attributes.txt')),
# ('CAMELS_NESTEDNESS_FILE', None),
# ('CAMELS_BASINS_SHP_FILE', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_cl/CAMELScl_catchment_boundaries/catchments_camels_cl_v1.3.shp'))])


def test_download_data_source():
    camelscl = CamelsCl(download=True)   # PASSED                      [100%]
