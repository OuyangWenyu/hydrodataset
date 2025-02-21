from hydrodataset import CamelsDe

def test_read_forcing():
    camelsde = CamelsDe()
    gage_ids = camelsde.read_object_ids()
    print(gage_ids)
    forcings = camelsde.read_relevant_cols(
        gage_ids[:5],
        ["1951-01-01", "2020-12-31"],
		var_lst = ["water_level", "precipitation_mean", "precipitation_min", "precipitation_median", "precipitation_max",
           "precipitation_stdev", "humidity_mean", "humidity_min", "humidity_median"]
    )
    print(forcings)

def test_read_streamflow():
    camelsde = CamelsDe()
    gage_ids = camelsde.read_object_ids()
    streamflow = camelsde.read_target_cols(
        gage_ids[:5],
        ["1951-01-01", "2020-12-31"],
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

def test_cache_xrdataset():
    camelsde = CamelsDe()
    cachexrdataset = camelsde.cache_xrdataset()


