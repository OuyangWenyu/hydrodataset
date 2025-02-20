from hydrodataset import CamelsCh

def test_read_forcing():
    camelsch = CamelsCh()
    gage_ids = camelsch.read_object_ids()
    print(gage_ids)
    forcings = camelsch.read_relevant_cols(
        gage_ids[:5],
        ["1981-01-01","2020-12-31"],
        var_lst=["waterlevel(m)", "precipitation(mm/d)", "temperature_min(degC)", "temperature_mean(degC)", "temperature_max(degC)", "rel_sun_dur(%)", "swe(mm)"]
    )
    print(forcings)

def test_read_streamflow():
    camelsch = CamelsCh()
    gage_ids = camelsch.read_object_ids()
    streamflow = camelsch.read_target_cols(
        gage_ids[:5],
        ["1981-01-01","2020-12-31"],
        target_cols=["discharge_vol(m3/s)", "discharge_spec(mm/d)"],
    )
    print(streamflow)

def test_read_attr():
    camelsch = CamelsCh()
    gage_ids = camelsch.read_object_ids()
    attributes = camelsch.read_constant_cols(
        gage_ids[:5],
        ["area","p_mean","crop_perc"]
    )
    print(attributes)

def test_cache_forcing():
    camelsch = CamelsCh()
    cacheforcing = camelsch.cache_forcing_xrdataset()

