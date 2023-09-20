"""
Author: Wenyu Ouyang
Date: 2022-09-05 23:20:24
LastEditTime: 2023-09-20 15:48:07
LastEditors: Wenyu Ouyang
Description: Tests for `hydrodataset` package
FilePath: \hydrodataset\tests\test_camels.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import io
import sys
import async_retriever as ar

from hydrodataset import CACHE_DIR
from hydrodataset import Camels


def test_binary():
    urls = ["https://gdex.ucar.edu/dataset/camels/file/basin_set_full_res.zip"]
    cache_name = CACHE_DIR.joinpath(urls[0].split("/")[-1] + ".sqlite")
    r = ar.retrieve(urls, "binary", cache_name=cache_name, ssl=False)
    r_b = ar.retrieve_binary(urls, cache_name=cache_name, ssl=False)
    save_zip = CACHE_DIR.joinpath(urls[0].split("/")[-1])
    save_zip.write_bytes(io.BytesIO(r[0]).getbuffer())
    assert sys.getsizeof(r[0]) == sys.getsizeof(r_b[0]) == 45179592


def test_stream():
    url = "https://gdex.ucar.edu/dataset/camels/file/basin_set_full_res.zip"
    temp_name = CACHE_DIR.joinpath("basin_set_full_res.zip")
    ar.stream_write([url], [temp_name])


def test_cache():
    camels = Camels()
    camels.cache_xrdataset()


def test_read_forcing():
    camels = Camels()
    gage_ids = camels.read_object_ids()
    forcings = camels.read_relevant_cols(
        gage_ids[:5], ["1980-01-01", "2015-01-01"], var_lst=["dayl", "prcp", "PET"]
    )
    print(forcings)


def test_read_tsxrdataset():
    camels = Camels()
    gage_ids = camels.read_object_ids()
    ts_data = camels.read_ts_xrdataset(
        gage_id_lst=gage_ids[:5],
        t_range=["2013-01-01", "2014-01-01"],
        var_lst=["streamflow"],
    )
    print(ts_data)


def test_read_attr_xrdataset():
    camels = Camels()
    gage_ids = camels.read_object_ids()
    attr_data = camels.read_attr_xrdataset(
        gage_id_lst=gage_ids[:5],
        var_lst=["soil_conductivity", "elev_mean", "geol_1st_class"],
        all_number=True,
    )
    print(attr_data)


def test_read_mean_prcp():
    camels = Camels()
    gage_ids = camels.read_object_ids()
    mean_prcp = camels.read_mean_prcp(gage_ids[:5])
    print(mean_prcp)
