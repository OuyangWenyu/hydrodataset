from hydrodataset import CamelsSe

def test_read_forcing():
    camelsse = CamelsSe()
    gage_ids = camelsse.read_object_ids()
    print(gage_ids)
    forcings = camelsse.read_relevant_cols(
        gage_ids[:5],
        ["1961-01-01", "2021-01-01"],
		var_lst = ["Pobs_mm", "Tobs_C"]
    )
    print(forcings)

def test_read_streamflow():
    camelsse = CamelsSe()
    gage_ids = camelsse.read_object_ids()
    streamflow = camelsse.read_target_cols(
        gage_ids[:5],
        ["1961-01-01", "2021-01-01"],
		target_cols = ["Qobs_m3s", "Qobs_mm"],
    )
    print(streamflow)

def test_read_attr():
    camelsse = CamelsSe()
    gage_ids = camelsse.read_object_ids()
    attributes = camelsse.read_constant_cols(
        gage_ids[:5],
        ["Area_km2", "Pmean_mm_year", "Wetlands_percentage"]
    )
    print(attributes)

def test_cache_forcing():
    camelsse = CamelsSe()
    cacheforcing = camelsse.cache_forcing_xrdataset()

def test_cache_streamflow():
    camelsse = CamelsSe()
    cachestreamflow = camelsse.cache_streamflow_xrdataset()

def test_cache_attributes():
    camelsse = CamelsSe()
    cacheatributes = camelsse.cache_attributes_xrdataset()

def test_cache_xrdataset():
    camelsse = CamelsSe()
    cachexrdataset = camelsse.cache_xrdataset()

def test_read_area_meanprcp():
    camelsse = CamelsSe()
    gage_ids = camelsse.read_object_ids()
    areas = camelsse.read_area(gage_ids[:5])
    mean_prcp = camelsse.read_mean_prcp(gage_ids[:5])
    print(areas.values)
    print(mean_prcp.values)

def test_set_data_source_describe():
    camelsSe = CamelsSe()
    describle = camelsSe.set_data_source_describe()
    print("\n")
    print(describle)
# OrderedDict([('CAMELS_DIR', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_se')),
# ('CAMELS_FLOW_DIR', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_se/catchment time series/catchment time series')),
# ('CAMELS_FORCING_DIR', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_se/catchment time series/catchment time series')),
# ('CAMELS_FORCING_TYPE', ['obs']),
# ('CAMELS_ATTR_DIR', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_se/catchment properties/catchment properties')),
# ('CAMELS_ATTR_KEY_LST', ['hydrological_signatures_1961_2020', 'landcover', 'physical_properties', 'soil_classes']),
# ('CAMELS_GAUGE_FILE', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_se/catchment properties/catchment properties/catchments_physical_properties.csv')),
# ('CAMELS_NESTEDNESS_FILE', None),
# ('CAMELS_BASINS_SHP', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_se/catchment_GIS_shapefiles/catchment_GIS_shapefiles/Sweden_catchments_50_boundaries_WGS84.shp'))])
