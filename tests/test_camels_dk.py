from hydrodataset import CamelsDk

def test_read_forcing():
    camelsdk = CamelsDk()
    gage_ids = camelsdk.read_object_ids()
    print(gage_ids)
    forcings = camelsdk.read_relevant_cols(
        gage_ids[:5],
        ["1989-01-02", "2023-12-31"],
		var_lst = ["precipitation","temperature","pet","DKM_dtp","DKM_eta","DKM_wcr","DKM_sdr","DKM_sre","DKM_gwh","Qdkm","DKM_irr","Abstraction"]
    )
    print(forcings)

def test_read_streamflow():
    camelsdk = CamelsDk()
    gage_ids = camelsdk.read_object_ids()
    streamflow = camelsdk.read_target_cols(
        gage_ids[:5],
        ["1989-01-02", "2023-12-31"],
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



