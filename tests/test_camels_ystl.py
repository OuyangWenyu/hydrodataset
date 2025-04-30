
import numpy as np
import pandas as pd

from hydrodataset import CamelsYstl

def test_convert_6h21d():
    file = r"D:\minio\waterism\datasets-origin\camels\camels_ystl\camels_ystl_1000_6h.csv"
    data_temp = pd.read_csv(file, sep=",")
    var_lst = list(data_temp.columns[:])
    date_6h = data_temp[var_lst[0]].values.tolist()
    prce_6h = data_temp[var_lst[1]].values.tolist()
    pet_6h = data_temp[var_lst[2]].values.tolist()
    discharge_6h = data_temp[var_lst[3]].values.tolist()
    date = []
    prcp = []
    pet = []
    discharge = []
    n_6h = data_temp.shape[0]
    n_1d = int(n_6h/4)
    k = 0
    for i in range(n_1d):
        data_i = date_6h[k].split("-")[:3]
        data_i = data_i[0] + "-" + data_i[1] + "-" + data_i[2]
        prcp_i = (sum(prce_6h[k:k+4]))
        pet_i = sum(pet_6h[k:k+4])
        discharge_i = sum(discharge_6h[k:k+4])/4
        k = k + 4
        date.append(data_i)
        prcp.append(prcp_i)
        pet.append(pet_i)
        discharge.append(discharge_i)

    prcp = np.array(prcp)
    pet = np.array(pet)
    discharge = np.array(discharge)
    prcp = np.around(prcp, decimals=2)
    pet = np.around(pet, decimals=3)
    discharge = np.around(discharge, decimals=1)

    data_1d = pd.DataFrame({var_lst[0]: date, var_lst[1]: prcp, var_lst[2]: pet, var_lst[3]: discharge,})
    data_1d.set_index(var_lst[0], inplace=True)
    file_name = r"D:\minio\waterism\datasets-origin\camels\camels_ystl\camels_ystl_1000.csv"
    data_1d.to_csv(file_name, sep=",")


def test_read_forcing():
    camelsystl = CamelsYstl()
    gage_ids = camelsystl.gage
    print(gage_ids)
    forcings = camelsystl.read_relevant_cols(
        gage_ids[:],
        ["1990-01-01","1994-01-01"],
        var_lst=["prcp",
                "pet",
                "discharge_vol",]
    )
    print(forcings)

def test_read_streamflow():
    camelsystl = CamelsYstl()
    gage_ids = camelsystl.gage
    streamflow = camelsystl.read_target_cols(
        gage_ids[:],
        ["1990-01-01","1994-01-01"],
        target_cols=["discharge_vol"],
    )
    print(streamflow)

def test_read_attr():
    camelsystl = CamelsYstl()
    gage_ids = camelsystl.gage
    attributes = camelsystl.read_constant_cols(
        gage_ids[:],
        ["area",]
    )
    print(attributes)

def test_cache_forcing():
    camelsystl = CamelsYstl()
    cacheforcing = camelsystl.cache_forcing_xrdataset()

def test_cache_streamflow():
    camelsystl = CamelsYstl()
    cachestreamflow = camelsystl.cache_streamflow_xrdataset()

def test_cache_attributes():
    camelsystl = CamelsYstl()
    cacheatributes = camelsystl.cache_attributes_xrdataset()

def test_cache_xrdataset():
    camelsystl = CamelsYstl()
    cachexrdataset = camelsystl.cache_xrdataset()
    print(cachexrdataset)
    print("done!")
# ============================= test session starts =============================
# collecting ... collected 1 item
#
# test_camels_ystl.py::test_cache_xrdataset
#
# ======================== 1 passed, 2 warnings in 7.25s ========================
# PASSED                         [100%]None
# done!
#
# Process finished with exit code 0
