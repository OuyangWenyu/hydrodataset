from hydrodataset import CamelsGb

def test_read_forcing():
    camelsgb = CamelsGb()
    gage_ids = camelsgb.read_object_ids()
    print(gage_ids)
    forcings = camelsgb.read_relevant_cols(
        gage_ids[:5],
        ["1970-10-01", "2015-10-01"],
		var_lst = ["precipitation", "pet", "temperature", "peti", "humidity", "shortwave_rad", "longwave_rad", "windspeed"]
    )
    print(forcings)

def test_read_streamflow():
    camelsgb = CamelsGb()
    gage_ids = camelsgb.read_object_ids()
    streamflow = camelsgb.read_target_cols(
        gage_ids[:5],
        ["1970-10-01", "2015-10-01"],
		target_cols = ["discharge_spec", "discharge_vol",],
    )
    print(streamflow)

def test_read_attr():
    camelsgb = CamelsGb()
    gage_ids = camelsgb.read_object_ids()
    attributes = camelsgb.read_constant_cols(
        gage_ids[:5],
        ["p_mean", "area", "grass_perc"]
    )
    print(attributes)

def test_cache_forcing():
    camelsgb = CamelsGb()
    cacheforcing = camelsgb.cache_forcing_xrdataset()

def test_cache_streamflow():
    camelsgb = CamelsGb()
    cachestreamflow = camelsgb.cache_streamflow_xrdataset()

def test_cache_attributes():
    camelsgb = CamelsGb()
    cacheatributes = camelsgb.cache_attributes_xrdataset()

def test_cache_xrdataset():
    camelsgb = CamelsGb()
    cachexrdataset = camelsgb.cache_xrdataset()

def test_read_area_meanprcp():
    camelsgb = CamelsGb()
    gage_ids = camelsgb.read_object_ids()
    areas = camelsgb.read_area(gage_ids[:5])
    mean_prcp = camelsgb.read_mean_prcp(gage_ids[:5])
    print(areas.values)
    print(mean_prcp.values)

def test_set_data_source_describe():
    camelsgb = CamelsGb()
    describle = camelsgb.set_data_source_describe()
    print("\n")
    print(describle)
# OrderedDict([('CAMELS_DIR', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_gb')),
# ('CAMELS_FLOW_DIR', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_gb/8344e4f3-d2ea-44f5-8afa-86d2987543a9/8344e4f3-d2ea-44f5-8afa-86d2987543a9/data/timeseries')),
# ('CAMELS_FORCING_DIR', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_gb/8344e4f3-d2ea-44f5-8afa-86d2987543a9/8344e4f3-d2ea-44f5-8afa-86d2987543a9/data/timeseries')),
# ('CAMELS_ATTR_DIR', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_gb/8344e4f3-d2ea-44f5-8afa-86d2987543a9/8344e4f3-d2ea-44f5-8afa-86d2987543a9/data')),
# ('CAMELS_ATTR_KEY_LST', ['climatic', 'humaninfluence', 'hydrogeology', 'hydrologic', 'hydrometry', 'landcover', 'soil', 'topographic']),
# ('CAMELS_GAUGE_FILE',
# WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_gb/8344e4f3-d2ea-44f5-8afa-86d2987543a9/8344e4f3-d2ea-44f5-8afa-86d2987543a9/data/CAMELS_GB_hydrometry_attributes.csv')),
# ('CAMELS_NESTEDNESS_FILE', None),
# ('CAMELS_BASINS_SHP_FILE',
# WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_gb/8344e4f3-d2ea-44f5-8afa-86d2987543a9/8344e4f3-d2ea-44f5-8afa-86d2987543a9/data/CAMELS_GB_catchment_boundaries/CAMELS_GB_catchment_boundaries.shp'))])


def test_download_data_source():
    camelsgb = CamelsGb(download=True)  # PASSED                      [100%]
