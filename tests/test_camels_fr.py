from hydrodataset import CamelsFr

def test_read_forcing():
    camelsfr = CamelsFr()
    gage_ids = camelsfr.read_object_ids()
    print(gage_ids)
    forcings = camelsfr.read_relevant_cols(
        gage_ids[:5],
        ["1970-01-01", "2021-12-31"],
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
        ["1970-01-01", "2021-12-31"],    # todo: the test failed, for the date formate
        # ["19700101", "20211231"],
		target_cols = ["tsd_q_l", "tsd_q_mm"],
    )
    print(streamflow)

def test_read_attr():
    camelsfr = CamelsFr()
    gage_ids = camelsfr.read_object_ids()
    attributes = camelsfr.read_constant_cols(       #todo: the test failed, a bug
        gage_ids[:5],
        ["sit_area_hydro", "sol_sand", "hgl_thm_bedrock"]
    )
    print(attributes)

