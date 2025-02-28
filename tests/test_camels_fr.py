from hydrodataset import CamelsFr

def test_read_forcing():
    camelsfr = CamelsFr()
    gage_ids = camelsfr.read_object_ids()
    print(gage_ids)
    forcings = camelsfr.read_relevant_cols(
        gage_ids[:5],
        ["1970-01-01", "2022-01-01"],
		var_lst = ["tsd_prec","tsd_prec_solid_frac","tsd_temp","tsd_pet_ou","tsd_pet_pe","tsd_pet_pm","tsd_wind",
                   "tsd_humid","tsd_rad_dli","tsd_rad_ssi","tsd_swi_gr","tsd_swi_isba","tsd_swe_isba","tsd_temp_min",
                   "tsd_temp_max"]
    )
    print(forcings)

def test_read_streamflow():
    camelsfr = CamelsFr()
    gage_ids = camelsfr.read_object_ids()
    streamflow = camelsfr.read_target_cols(
        gage_ids[:5],
        ["1970-01-01", "2022-01-01"],
		target_cols = ["tsd_q_l", "tsd_q_mm"],
    )
    print(streamflow)

def test_read_attr():
    camelsfr = CamelsFr()
    gage_ids = camelsfr.read_object_ids()
    attributes = camelsfr.read_constant_cols(
        gage_ids[:5],
        ["geo_dom_class", "hgl_krs_karstic", "hgl_thm_bedrock"]
    )
    print(attributes)

def test_cache_forcing():
    camelsfr = CamelsFr()
    cacheforcing = camelsfr.cache_forcing_xrdataset()

def test_cache_streamflow():
    camelsfr = CamelsFr()
    cachestreamflow = camelsfr.cache_streamflow_xrdataset()

def test_cache_attributes():
    camelsfr = CamelsFr()
    cacheatributes = camelsfr.cache_attributes_xrdataset()

def test_cache_xrdataset():
    camelsfr = CamelsFr()
    cachexrdataset = camelsfr.cache_xrdataset()


