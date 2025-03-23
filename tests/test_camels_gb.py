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
