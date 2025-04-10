from hydrodataset import CamelsDe

def test_read_forcing():
    camelsde = CamelsDe()
    gage_ids = camelsde.read_object_ids()
    print(gage_ids)
    forcings = camelsde.read_relevant_cols(
        gage_ids[:5],
        ["1951-01-01", "2021-01-01"],
		var_lst = ["water_level", "precipitation_mean", "precipitation_min", "precipitation_median", "precipitation_max",
           "precipitation_stdev", "humidity_mean", "humidity_min", "humidity_median"]
    )
    print(forcings)

def test_read_streamflow():
    camelsde = CamelsDe()
    gage_ids = camelsde.read_object_ids()
    streamflow = camelsde.read_target_cols(
        gage_ids[:5],
        ["1951-01-01", "2021-01-01"],
		target_cols = ["discharge_vol", "discharge_spec"],
    )
    print(streamflow)

def test_read_attr():
    camelsde = CamelsDe()
    gage_ids = camelsde.read_object_ids()
    attributes = camelsde.read_constant_cols(
        gage_ids[:5],
        ["area", "p_mean", "dams_num"]
    )
    print(attributes)

def test_cache_forcing():
    camelsde = CamelsDe()
    cacheforcing = camelsde.cache_forcing_xrdataset()

def test_cache_streamflow():
    camelsde = CamelsDe()
    cachestreamflow = camelsde.cache_streamflow_xrdataset()

def test_cache_attributes():
    camelsde = CamelsDe()
    cacheatributes = camelsde.cache_attributes_xrdataset()

def test_cache_xrdataset():
    camelsde = CamelsDe()
    cachexrdataset = camelsde.cache_xrdataset()

def test_read_area_meanprcp():
    camelsde = CamelsDe()
    gage_ids = camelsde.read_object_ids()
    areas = camelsde.read_area(gage_ids[:5])
    mean_prcp = camelsde.read_mean_prcp(gage_ids[:5])
    print(areas.values)
    print(mean_prcp.values)

def test_set_data_source_describe():
    camelsde = CamelsDe()
    describle = camelsde.set_data_source_describe()
    print("\n")
    print(describle)
# OrderedDict([('CAMELS_DIR', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_de')),
# ('CAMELS_FLOW_DIR', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_de/timeseries')),
# ('CAMELS_FORCING_DIR', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_de/timeseries')),
# ('CAMELS_FORCING_TYPE', ['observation', 'simulated']),
# ('CAMELS_ATTR_DIR', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_de')),
# ('CAMELS_ATTR_KEY_LST', ['climatic', 'humaninfluence', 'hydrogeology', 'hydrologic', 'landcover', 'soil', 'topographic']),
# ('CAMELS_GAUGE_FILE', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_de/CAMELS_DE_hydrologic_attributes.csv')),
# ('CAMELS_NESTEDNESS_FILE', None),
# ('CAMELS_BASINS_SHP', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_de/CAMELS_DE_catchment_boundaries/catchments/CAMELS_DE_catchments.shp'))])
