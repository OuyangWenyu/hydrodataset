"""
Author: Wenyu Ouyang
Date: 2022-09-06 16:53:45
LastEditTime: 2022-09-08 19:28:44
LastEditors: Wenyu Ouyang
Description: util functions
FilePath: \hydrodataset\hydrodataset\hydro_utils.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import datetime as dt
import io
from pathlib import Path
from typing import Union
import zipfile
import numpy as np
import requests
import async_retriever as ar
from hydrodataset import CACHE_DIR


def download_one_zip(data_url: str, data_dir: str) -> None:
    """
    A normal way to download one zip file from url as data_file

    We recommend you to use async_retriever to download files

    Parameters
    ----------
    data_url
        the URL of the downloading website
    data_dir
        where we will put the data
    """

    r = requests.get(data_url, stream=True)
    with open(data_dir, "wb") as py_file:
        for chunk in r.iter_content(chunk_size=1024):  # 1024 bytes
            if chunk:
                py_file.write(chunk)


def download_zip_files(urls, the_dir: Path):
    """Download multi-files from multi-urls

    Parameters
    ----------
    urls : list
        list of all urls
    the_dir : Path
        the directory containing all downloaded files
    """
    cache_names = CACHE_DIR.joinpath(the_dir.stem + ".sqlite")
    r = ar.retrieve(urls, "binary", cache_name=cache_names, ssl=False)
    files = [the_dir.joinpath(url.split("/")[-1]) for url in urls]
    [files[i].write_bytes(io.BytesIO(r[i]).getbuffer()) for i in range(len(files))]


def zip_extract(the_dir) -> None:
    """Extract the downloaded zip files in the_dir"""
    for f in the_dir.glob("*.zip"):
        with zipfile.ZipFile(f) as zf:
            # extract files to a directory named by f.stem
            zf.extractall(the_dir.joinpath(f.stem))


def t_range_days(t_range, *, step=np.timedelta64(1, "D")):
    sd = dt.datetime.strptime(t_range[0], "%Y-%m-%d")
    ed = dt.datetime.strptime(t_range[1], "%Y-%m-%d")
    t_array = np.arange(sd, ed, step)
    return t_array


def t2str(t_: Union[str, dt.datetime]):
    if type(t_) is str:
        t_str = dt.datetime.strptime(t_, "%Y-%m-%d")
        return t_str
    elif type(t_) is dt.datetime:
        t = t_.strftime("%Y-%m-%d")
        return t
    else:
        raise NotImplementedError("We don't support this data type yet")
