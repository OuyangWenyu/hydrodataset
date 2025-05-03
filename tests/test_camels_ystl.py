
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
    discharge1 = []
    discharge2 = []
    discharge3 = []
    n_6h = data_temp.shape[0]
    n_1d = int(n_6h/4)
    k = 0
    for i in range(n_1d):
        data_i = date_6h[k].split("-")[:3]
        data_i = data_i[0] + "-" + data_i[1] + "-" + data_i[2]
        prcp_i = sum(prce_6h[k:k+4])
        pet_i = sum(pet_6h[k:k+4])
        # discharge_i = sum(discharge_6h[k:k+4])/4
        discharge_i = discharge_6h[k]
        discharge_i1 = discharge_6h[k+1]
        discharge_i2 = discharge_6h[k+2]
        discharge_i3 = discharge_6h[k+3]
        k = k + 4
        date.append(data_i)
        prcp.append(prcp_i)
        pet.append(pet_i)
        discharge.append(discharge_i)
        discharge1.append(discharge_i1)
        discharge2.append(discharge_i2)
        discharge3.append(discharge_i3)

    prcp = np.array(prcp)
    pet = np.array(pet)
    discharge = np.array(discharge)
    prcp = np.around(prcp, decimals=2)
    pet = np.around(pet, decimals=3)
    discharge = np.around(discharge, decimals=1)
    discharge1 = np.around(discharge1, decimals=1)
    discharge2 = np.around(discharge2, decimals=1)
    discharge3 = np.around(discharge3, decimals=1)

    data_1d = pd.DataFrame({var_lst[0]: date, var_lst[1]: prcp, var_lst[2]: pet, var_lst[3]: discharge, "discharge_vol1": discharge1, "discharge_vol2": discharge2, "discharge_vol3": discharge3})
    data_1d.set_index(var_lst[0], inplace=True)
    # data_1d.drop(axis=0, index="1992-02-29", inplace=True)
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
                 "discharge_vol1",
                 "discharge_vol2",
                 "discharge_vol3",
                 ]
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
# ======================== 1 passed, 2 warnings in 3.46s ========================
# PASSED                         [100%]None
# done!
# Read streamflow data of CAMELS-YSTL:   0%|          | 0/1 [00:00<?, ?it/s]
# 100%|██████████| 1/1 [00:00<00:00, 111.08it/s]
# Read streamflow data of CAMELS-YSTL: 100%|██████████| 1/1 [00:00<00:00, 99.99it/s]
# Read forcing data of CAMELS-YSTL:   0%|          | 0/3 [00:00<?, ?it/s]
# 100%|██████████| 1/1 [00:00<00:00, 200.01it/s]
#
# 100%|██████████| 1/1 [00:00<00:00, 249.97it/s]
#
# 100%|██████████| 1/1 [00:00<00:00, 243.46it/s]
# Read forcing data of CAMELS-YSTL: 100%|██████████| 3/3 [00:00<00:00, 212.66it/s]
