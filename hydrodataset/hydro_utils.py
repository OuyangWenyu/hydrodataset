"""
Author: Wenyu Ouyang
Date: 2022-09-06 16:53:45
LastEditTime: 2023-07-30 20:09:20
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
import xarray as xr


def map_string_vars(ds):
    # Iterate over all variables in the dataset
    for var in ds.data_vars:
        # Check if the variable contains string data
        if ds[var].dtype == object:
            # Convert the DataArray to a pandas Series
            var_series = ds[var].to_series()

            # Get all unique strings and create a mapping to integers
            unique_strings = sorted(var_series.unique())
            mapping = {value: i for i, value in enumerate(unique_strings)}

            # Apply the mapping to the series
            mapped_series = var_series.map(mapping)

            # Convert the series back to a DataArray and replace the old one in the Dataset
            ds[var] = xr.DataArray(mapped_series)

    return ds


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
    cache_names = CACHE_DIR.joinpath(f"{the_dir.stem}.sqlite")
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
    return np.arange(sd, ed, step)


def t2str(t_: Union[str, dt.datetime]):
    if type(t_) is str:
        return dt.datetime.strptime(t_, "%Y-%m-%d")
    elif type(t_) is dt.datetime:
        return t_.strftime("%Y-%m-%d")
    else:
        raise NotImplementedError("We don't support this data type yet")
