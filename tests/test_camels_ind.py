from hydrodataset import CamelsInd


def test_read_forcing():
    camelsind = CamelsInd()
    gage_ids = camelsind.read_object_ids()
    print(gage_ids)
    forcings = camelsind.read_relevant_cols(
        gage_ids[:5],
        ["1980-01-01", "2021-01-01"],
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
        ["1980-01-01", "2021-01-01"],
        target_cols=["streamflow_observed"],
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

def test_cache_forcing():
    camelsind = CamelsInd()
    cacheforcing = camelsind.cache_forcing_xrdataset()

def test_cache_streamflow():
    camelsind = CamelsInd()
    cachestreamflow = camelsind.cache_streamflow_xrdataset()

def test_cache_attributes():
    camelsind = CamelsInd()
    cacheatributes = camelsind.cache_attributes_xrdataset()

def test_cache_xrdataset():
    camelsind = CamelsInd()
    cachexrdataset = camelsind.cache_xrdataset()

def test_read_area_meanprcp():
    camelsind = CamelsInd()
    gage_ids = camelsind.read_object_ids()
    areas = camelsind.read_area(gage_ids[:5])
    mean_prcp = camelsind.read_mean_prcp(gage_ids[:5])
    print(areas.values)
    print(mean_prcp.values)

def test_set_data_source_describe():
    camelsind = CamelsInd()
    describle = camelsind.set_data_source_describe()
    print("\n")
    print(describle)
# OrderedDict([('CAMELS_DIR', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_ind')),
# ('CAMELS_FLOW_DIR', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_ind/CAMELS_IND_All_Catchments/streamflow_timeseries/streamflow_observed.csv')),
# ('CAMELS_FORCING_DIR', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_ind/CAMELS_IND_All_Catchments/catchment_mean_forcings')),
# ('CAMELS_FORCING_TYPE', ['observation']),
# ('CAMELS_ATTR_DIR', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_ind/CAMELS_IND_All_Catchments/attributes_csv')),
# ('CAMELS_ATTR_KEY_LST', ['anth', 'clim', 'geol', 'hydro', 'land', 'name', 'soil', 'topo']),
# ('CAMELS_GAUGE_FILE', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_ind/CAMELS_IND_All_Catchments/attributes_csv/camels_ind_clim.csv')),
# ('CAMELS_NESTEDNESS_FILE', None),
# ('CAMELS_BASINS_SHP', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_ind/CAMELS_IND_All_Catchments/shapefiles_catchment/merged/all_catchments.shp'))])


def test_download_data_source():
    camelsind = CamelsInd(download=True)
