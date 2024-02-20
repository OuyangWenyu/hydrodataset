"""
Author: Wenyu Ouyang
Date: 2024-02-18 20:42:07
LastEditTime: 2024-02-20 13:16:55
LastEditors: Wenyu Ouyang
Description: Test cases for `hydrodataset.camelshourly` module
FilePath: \hydrodataset\tests\test_camelshourly.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import pytest
import os
import pandas as pd
import xarray as xr
import numpy as np
import tempfile
import timeit
from hydrodataset.camelshourly import CamelsHourly


@pytest.fixture()
def camels_hourly():
    data_path = os.path.join("camels", "camels_us_hourly")
    download = True
    region = "US"

    return CamelsHourly(data_path, download, region)


def test_get_name(camels_hourly):
    expected_name = "HOURLY_CAMELS_US"

    result = camels_hourly.get_name()

    assert result == expected_name


def test_read_relevant_cols(camels_hourly):
    gage_id_lst = ["01022500", "09447800", "14400000"]
    t_range = ["1995-01-01", "2005-01-01"]
    var_lst = ["potential_evaporation", "total_precipitation"]

    # Test when var_lst is None
    result = camels_hourly.read_relevant_cols(
        gage_id_lst=gage_id_lst, t_range=t_range, var_lst=None
    )

    # Test when var_lst contains variables not in forcing.columns
    with pytest.raises(ValueError):
        camels_hourly.read_relevant_cols(
            gage_id_lst=gage_id_lst,
            t_range=t_range,
            var_lst=["streamflow", "humidity"],
        )

    # Test when var_lst is valid
    result = camels_hourly.read_relevant_cols(
        gage_id_lst=gage_id_lst, t_range=t_range, var_lst=var_lst
    )
    start_date = t_range[0]
    # the given range is left-closed and right-open
    end_date = pd.to_datetime(t_range[1]) - pd.Timedelta(hours=1)
    t_range_lst = pd.date_range(start=start_date, end=end_date, freq="H")
    assert result.shape == (3, len(t_range_lst), len(var_lst))


def test_read_target_cols(camels_hourly):
    gage_id_lst = ["01022500", "09447800", "14400000"]
    t_range = ["1995-01-01", "2005-01-01"]
    target_cols = ["qobs_mm_per_hour"]

    # Test when target_cols is None
    result = camels_hourly.read_target_cols(
        gage_id_lst=gage_id_lst, t_range=t_range, target_cols=None
    )
    assert result is None

    # Test when target_cols contains variables not in self.get_target_cols()
    with pytest.raises(ValueError):
        camels_hourly.read_target_cols(
            gage_id_lst=gage_id_lst,
            t_range=t_range,
            target_cols=["temperature", "humidity"],
        )

    # Test when target_cols is valid
    expected_t_range = pd.date_range(start=t_range[0], end=t_range[1], freq="H")

    result = camels_hourly.read_target_cols(
        gage_id_lst=gage_id_lst, t_range=t_range, target_cols=target_cols
    )
    # the given range is left-closed and right-open
    assert result.shape == (3, len(expected_t_range) - 1, len(target_cols))


def create_nc_file_with_xarray(filename, shape=(460, 460, 460)):
    # 460^3 elements * 8 bytes/element ≈ 100MB
    data = np.random.rand(*shape)
    coords = {dim: range(size) for dim, size in zip(["x", "y", "z"], shape)}
    ds = xr.Dataset({"data": (list(coords.keys()), data)}, coords=coords)
    ds.to_netcdf(filename)
    return data  # 返回原始数据以便于后续比较


def convert_nc_to_zarr(nc_filename, zarr_filename):
    ds = xr.open_dataset(nc_filename, engine="netcdf4")
    ds.to_zarr(zarr_filename)


def test_create_and_convert_nc_file():
    fd, nc_temp_file = tempfile.mkstemp(suffix=".nc")
    try:
        original_data = create_nc_file_with_xarray(nc_temp_file)

        with tempfile.TemporaryDirectory() as tmp_zarr_dir:
            convert_nc_to_zarr(nc_temp_file, tmp_zarr_dir)

            zarr_data = xr.open_zarr(tmp_zarr_dir)
            assert os.path.exists(tmp_zarr_dir)
            assert os.path.isdir(tmp_zarr_dir)

            np.testing.assert_array_equal(original_data, zarr_data["data"].values)
    finally:
        os.close(fd)  # 先关闭文件
        os.remove(nc_temp_file)  # 然后删除文件


def test_read_performance():
    fd, nc_temp_file = tempfile.mkstemp(suffix=".nc")
    try:
        create_nc_file_with_xarray(nc_temp_file)

        with tempfile.TemporaryDirectory() as tmp_zarr_dir:
            convert_nc_to_zarr(nc_temp_file, tmp_zarr_dir)

            read_number = 2
            # 测量读取NC文件的时间
            nc_read_time = timeit.timeit(
                lambda: xr.open_dataset(nc_temp_file, engine="netcdf4"),
                number=read_number,
            )
            # 测量读取Zarr文件的时间
            zarr_read_time = timeit.timeit(
                lambda: xr.open_zarr(tmp_zarr_dir), number=read_number
            )

            print(f"Average time to read NC file: {nc_read_time / read_number:.5f} seconds")
            print(f"Average time to read Zarr file: {zarr_read_time / read_number:.5f} seconds")

            # 断言测试：可根据需要添加，例如比较读取时间
            # assert nc_read_time < zarr_read_time
    finally:
        os.close(fd)  # 先关闭文件
        os.remove(nc_temp_file)  # 然后删除文件


def test_cache_xrdataset(camels_hourly):
    # Test the caching of xr.Dataset
    camels_hourly.cache_xrdataset()

    # Verify that the zarr file exists
    zarr_file_path = (
        camels_hourly.data_source_description["CAMELS_US_HOURLY_TS_NC"][:-3] + "zarr"
    )
    assert os.path.exists(zarr_file_path)

    # Verify that the zarr file can be loaded as xr.Dataset
    ds = xr.open_zarr(zarr_file_path)
    assert isinstance(ds, xr.Dataset)


def test_read_ts_xrdataset(camels_hourly):
    gage_id_lst = ["01022500", "09447800", "14400000"]
    t_range = ["1995-01-01", "2005-01-01"]
    var_lst = ["qobs_mm_per_hour", "total_precipitation"]

    # Test when var_lst is None
    result = camels_hourly.read_ts_xrdataset(
        gage_id_lst=gage_id_lst, t_range=t_range, var_lst=None
    )
    assert result is None

    # Test when var_lst contains variables not in ts.variables
    # with pytest.raises(ValueError):
    #     camels_hourly.read_ts_xrdataset(
    #         gage_id_lst=gage_id_lst,
    #         t_range=t_range,
    #         var_lst=["streamflow", "temperature"],
    #     )

    # Test when var_lst is valid
    expected_t_range = pd.date_range(start=t_range[0], end=t_range[1], freq="H")

    result = camels_hourly.read_ts_xrdataset(
        gage_id_lst=gage_id_lst, t_range=t_range, var_lst=var_lst
    )

    assert isinstance(result, xr.Dataset)
    assert result.variables.keys() == var_lst
    assert result["streamflow"].shape == (48, 3)
    assert result["precipitation"].shape == (48, 3)
    assert result["streamflow"].coords["time"].equals(expected_t_range)
    assert result["streamflow"].coords["basin"].equals(pd.Index(gage_id_lst))
    assert result["precipitation"].coords["time"].equals(expected_t_range)
    assert result["precipitation"].coords["basin"].equals(pd.Index(gage_id_lst))
