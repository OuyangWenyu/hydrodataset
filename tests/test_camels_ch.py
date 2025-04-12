from hydrodataset import CamelsCh

def test_read_forcing():
    camelsch = CamelsCh()
    gage_ids = camelsch.read_object_ids()
    print(gage_ids)
    forcings = camelsch.read_relevant_cols(
        gage_ids[:5],
        ["1981-01-01","2021-01-01"],
        var_lst=["waterlevel(m)",
                 "precipitation(mm/d)",
                 "temperature_min(degC)",
                 "temperature_mean(degC)",
                 "temperature_max(degC)",
                 "rel_sun_dur(%)",
                 "swe(mm)",]
    )
    print(forcings)

def test_read_streamflow():
    camelsch = CamelsCh()
    gage_ids = camelsch.read_object_ids()
    streamflow = camelsch.read_target_cols(
        gage_ids[:5],
        ["1981-01-01","2021-01-01"],
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

def test_cache_streamflow():
    camelsch = CamelsCh()
    cachestreamflow = camelsch.cache_streamflow_xrdataset()

def test_cache_attributes():
    camelsch = CamelsCh()
    cacheatributes = camelsch.cache_attributes_xrdataset()

def test_cache_xrdataset():
    camelsch = CamelsCh()
    cachexrdataset = camelsch.cache_xrdataset()

def test_read_area_meanprcp():
    camelsch = CamelsCh()
    gage_ids = camelsch.read_object_ids()
    areas = camelsch.read_area(gage_ids[:5])
    mean_prcp = camelsch.read_mean_prcp(gage_ids[:5])
    print(areas.area.data)
    print(mean_prcp.p_mean.data)


def test_set_data_source_describe():
    camelsch = CamelsCh()
    describle = camelsch.set_data_source_describe()
    print("\n")
    print(describle)
    # OrderedDict([('CAMELS_DIR', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_ch')), (
    # 'CAMELS_FLOW_DIR', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_ch/timeseries/observation_based')),
    # ('CAMELS_FORCING_DIR',WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_ch/timeseries/observation_based')),
    # ('CAMELS_FORCING_TYPE', ['observation', 'simulation']), ('CAMELS_ATTR_DIR', WindowsPath(
    # 'D:/minio/waterism/datasets-origin/camels/camels_ch/static_attributes')),
    # ('CAMELS_ATTR_KEY_LST',['climate', 'geology', 'glacier','humaninfluence','hydrogeology', 'hydrology','landcover', 'soil','topographic', 'catchment']),
    # ('CAMELS_GAUGE_FILE', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_ch/static_attributes/CAMELS_CH_hydrology_attributes_obs.csv')),
    # ('CAMELS_NESTEDNESS_FILE', None), ('CAMELS_BASINS_SHP', WindowsPath(
    # 'D:/minio/waterism/datasets-origin/camels/camels_ch/catchment_delineations/CAMELS_CH_sub_catchments.shp'))])

def test_download_data_source():
    camelsch = CamelsCh(download=True)
