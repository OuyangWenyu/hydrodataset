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
