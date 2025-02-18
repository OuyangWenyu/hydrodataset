from hydrodataset import CamelsInd

# todo: the test for forcing and streamflow failed, for the date formate and the separation of streamflow and forcing data.
def test_read_forcing():
    camelsind = CamelsInd()
    gage_ids = camelsind.read_object_ids()
    print(gage_ids)
    forcings = camelsind.read_relevant_cols(
        gage_ids[:5],
        ["1980-01-01", "2020-12-31"],
        # ["1980,1,1", "2020,12,31"],       # todo: ?
        var_lst=["prcp(mm/day)", "tmax(C)", "tmin(C)", "tavg(C)", "srad_lw(w/m2)", "srad_sw(w/m2)", "wind_u(m/s)",
            "wind_v(m/s)", "wind(m/s)", "rel_hum(%)", "pet(mm/day)", "pet_gleam(mm/day)", "aet_gleam(mm/day)", "evap_canopy(kg/m2/s)",
            "evap_surface(kg/m2/s)", "sm_lvl1(kg/m2)", "sm_lvl2(kg/m2)", "sm_lvl3(kg/m2)", "sm_lvl4(kg/m2)"]
    )
    print(forcings)

def test_read_streamflow():
    camelsind = CamelsInd()
    gage_ids = camelsind.read_object_ids()
    streamflow = camelsind.read_target_cols(
        gage_ids[:5],
        ["1980-01-01", "2020-12-31"],
        # ["1980,1,1", "2020,12,31"],
        target_cols=["discharge_vol(m3/s)"],   # todo: ?
    )
    print(streamflow)

def test_read_attr():
    camelsind = CamelsInd()
    gage_ids = camelsind.read_object_ids()
    attributes = camelsind.read_constant_cols(
        gage_ids[:5],
        ["cwc_area","p_mean","soil_conductivity_top"]
    )
    print(attributes)

