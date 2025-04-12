from hydrodataset import CamelsAus

def test_read_forcing():
    camelaus = CamelsAus()
    gage_ids = camelaus.read_object_ids()
    print(gage_ids)
    forcings = camelaus.read_relevant_cols(
        gage_ids[:5],
        ["1990-01-01", "2010-01-01"],
		var_lst = ["precipitation_AWAP", "precipitation_SILO", "precipitation_var_AWAP", "et_morton_actual_SILO", "et_morton_point_SILO",
                   "et_morton_wet_SILO", "et_short_crop_SILO", "et_tall_crop_SILO", "evap_morton_lake_SILO", "evap_pan_SILO", "evap_syn_SILO",
                   "solarrad_AWAP", "tmax_AWAP", "tmin_AWAP", "vprp_AWAP", "mslp_SILO", "radiation_SILO", "rh_tmax_SILO", "rh_tmin_SILO", "tmax_SILO",
                   "tmin_SILO", "vp_deficit_SILO", "vp_SILO", ]
    )
    print(forcings)

def test_read_streamflow():
    camelaus = CamelsAus()
    gage_ids = camelaus.read_object_ids()
    streamflow = camelaus.read_target_cols(
        gage_ids[:5],
        ["1990-01-01", "2010-01-01"],
		target_cols = ["streamflow_MLd", "streamflow_MLd_inclInfilled", "streamflow_mmd"]
    )
    print(streamflow)

def test_read_attr():
    camelaus = CamelsAus()
    gage_ids = camelaus.read_object_ids()
    attributes = camelaus.read_constant_cols(
        gage_ids[:5],
        ["p_mean", "catchment_area", "baseflow_index"]
    )
    print(attributes)

def test_cache_forcing():
    camelaus = CamelsAus()
    cacheforcing = camelaus.cache_forcing_xrdataset()

def test_cache_streamflow():
    camelaus = CamelsAus()
    cachestreamflow = camelaus.cache_streamflow_xrdataset()

def test_cache_attributes():
    camelaus = CamelsAus()
    cacheatributes = camelaus.cache_attributes_xrdataset()

def test_cache_xrdataset():
    camelaus = CamelsAus()
    cachexrdataset = camelaus.cache_xrdataset()


def test_read_area_meanprcp():
    camelsaus = CamelsAus()
    gage_ids = camelsaus.read_object_ids()
    areas = camelsaus.read_area(gage_ids[:5])
    mean_prcp = camelsaus.read_mean_prcp(gage_ids[:5])
    print(areas.values)
    print(mean_prcp.values)

def test_download_data_source():
    camelsaus = CamelsAus(download=True)
