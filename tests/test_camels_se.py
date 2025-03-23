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
