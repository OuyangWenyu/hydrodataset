from hydrodataset import CamelsSe

def test_read_forcing():
    camelsse = CamelsSe()
    gage_ids = camelsse.read_object_ids()
    print(gage_ids)
    forcings = camelsse.read_relevant_cols(     # todo: do not pass the test
        gage_ids[:5],
        ["1961-01-01", "2020-12-31"],
		var_lst = ["Pobs_mm", "Tobs_C"]
    )
    print(forcings)

def test_read_streamflow():
    camelsse = CamelsSe()
    gage_ids = camelsse.read_object_ids()
    streamflow = camelsse.read_target_cols(     # todo: do not pass the test
        gage_ids[:5],
        ["1961-01-01", "2020-12-31"],
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

