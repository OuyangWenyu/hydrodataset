"""
Author: Wenyu Ouyang
Date: 2022-09-05 23:20:24
LastEditTime: 2022-10-05 17:58:41
LastEditors: Wenyu Ouyang
Description: Tests for `hydrodataset` package
FilePath: \hydrodataset\test\test_camels.py
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
    camels.cache_forcing_xrdataset()
