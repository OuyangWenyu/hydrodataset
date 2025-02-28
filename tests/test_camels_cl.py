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
