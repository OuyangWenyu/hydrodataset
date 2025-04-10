from hydrodataset import CamelsDk

def test_read_forcing():
    camelsdk = CamelsDk()
    gage_ids = camelsdk.read_object_ids()
    print(gage_ids)
    forcings = camelsdk.read_relevant_cols(
        gage_ids[:5],
        ["1989-01-02", "2024-01-01"],
		var_lst = ["precipitation","temperature","pet","DKM_dtp","DKM_eta","DKM_wcr","DKM_sdr","DKM_sre","DKM_gwh","DKM_irr","Abstraction"]
    )
    print(forcings)

def test_read_streamflow():
    camelsdk = CamelsDk()
    gage_ids = camelsdk.read_object_ids()
    streamflow = camelsdk.read_target_cols(
        gage_ids[:5],
        ["1989-01-02", "2024-01-01"],
		target_cols = ["Qobs"],
    )
    print(streamflow)

def test_read_attr():
    camelsdk = CamelsDk()
    gage_ids = camelsdk.read_object_ids()
    attributes = camelsdk.read_constant_cols(
        gage_ids[:5],
        ["catch_area", "p_mean", "slope_median"]
    )
    print(attributes)

def test_cache_forcing():
    camelsdk = CamelsDk()
    cacheforcing = camelsdk.cache_forcing_xrdataset()

def test_cache_streamflow():
    camelsdk = CamelsDk()
    cachestreamflow = camelsdk.cache_streamflow_xrdataset()

def test_cache_attributes():
    camelsdk = CamelsDk()
    cacheatributes = camelsdk.cache_attributes_xrdataset()

def test_cache_xrdataset():
    camelsdk = CamelsDk()
    cachexrdataset = camelsdk.cache_xrdataset()

def test_read_area_meanprcp():
    camelsdk = CamelsDk()
    gage_ids = camelsdk.read_object_ids()
    areas = camelsdk.read_area(gage_ids[:5])
    mean_prcp = camelsdk.read_mean_prcp(gage_ids[:5])
    print(areas.values)
    print(mean_prcp.values)

def test_read_gage_id():
    camelsdk = CamelsDk()
    gage_ids = camelsdk.gage
    print(gage_ids)


def test_read_gauge_id():
    """
    read the gages id of gauged catchments
    """
    import os
    camels_file = "D:\minio\waterism\datasets-origin\camels\camels_dk\Dynamics\Gauged_catchments"
    filename = os.listdir(camels_file)
    site_list = []
    for name in filename:
        name_ = name.split("_")[-1]
        site = name_.split(".")[0]
        site_list.append(site)
    print(site_list)

def test_set_data_source_describe():
    camelsdk = CamelsDk()
    describle = camelsdk.set_data_source_describe()
    print("\n")
    print(describle)
# OrderedDict([('CAMELS_DIR', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_dk')),
# ('CAMELS_FLOW_DIR', [WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_dk/Dynamics/Gauged_catchments'),
# WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_dk/Dynamics/Ungauged_catchments')]),
# ('CAMELS_FORCING_DIR', [WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_dk/Dynamics/Gauged_catchments'),
# WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_dk/Dynamics/Ungauged_catchments')]),
# ('CAMELS_ATTR_DIR', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_dk/Attributes')),
# ('CAMELS_ATTR_KEY_LST', ['climate', 'geology', 'landuse', 'signature_obs_based', 'soil', 'topography']),
# ('CAMELS_GAUGE_FILE', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_dk/Attributes/CAMELS_DK_climate.csv')),
# ('CAMELS_NESTEDNESS_FILE', None),
# ('CAMELS_BASINS_SHP', WindowsPath('D:/minio/waterism/datasets-origin/camels/camels_dk/Shapefile/CAMELS_DK_304_gauging_catchment_boundaries.shp'))])




