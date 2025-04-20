"""
Author: Wenyu Ouyang
Date: 2022-09-05 23:20:24
LastEditTime: 2024-11-11 17:29:13
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
import numpy as np
import pandas as pd
import xarray as xr
from unittest.mock import patch, MagicMock


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
        var_lst=["streamflow", "ET"],
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
# PASSED                          [100%]
# <xarray.Dataset> Size: 160B
# Dimensions:            (basin: 5)
# Coordinates:
#   * basin              (basin) object 40B '01013500' '01022500' ... '01047000'
# Data variables:
#     soil_conductivity  (basin) float64 40B 1.107 2.375 1.29 1.373 2.615
#     elev_mean          (basin) float64 40B 250.3 92.68 143.8 247.8 310.4
#     geol_1st_class     (basin) float64 40B 10.0 0.0 10.0 10.0 7.0


def test_read_mean_prcp():
    camels = Camels()
    gage_ids = camels.read_object_ids()
    mean_prcp = camels.read_mean_prcp(gage_ids[:5])
    print(mean_prcp)
    assert isinstance(mean_prcp, xr.Dataset)


def test_read_target_cols_us():
    camels = Camels()
    camels.region = "US"
    camels.read_camels_us_model_output_data = MagicMock(
        return_value=np.arange(1, 3653).reshape(3652, 1)
    )
    camels._read_augmented_camels_streamflow = MagicMock(
        return_value=np.arange(1, 3653)
    )

    gage_id_lst = ["01013500"]
    t_range = ["1990-01-01", "2000-01-01"]
    target_cols = ["usgsFlow", "ET"]

    result = camels.read_target_cols(gage_id_lst, t_range, target_cols)
    assert result.shape == (1, 3652, 2)


# TODO: THE FOLLOWING TESTS ARE NOT FULLY TESTED
def test_read_target_cols_aus():
    camels = Camels()
    camels.region = "AUS"
    camels.data_source_description = {"CAMELS_FLOW_DIR": "/path/to/flow_dir"}

    gage_id_lst = ["12345678"]
    t_range = ["1990-01-01", "2000-01-01"]
    target_cols = ["streamflow_mmd"]

    with patch("pandas.read_csv") as mock_read_csv:
        mock_df = pd.DataFrame(
            {
                "year": [1990, 1990, 1990],
                "month": [1, 1, 1],
                "day": [1, 2, 3],
                "12345678": [1.0, 2.0, 3.0],
            }
        )
        mock_read_csv.return_value = mock_df

        result = camels.read_target_cols(gage_id_lst, t_range, target_cols)
        assert result.shape == (1, 3653, 1)
        assert np.all(result[:, :3, 0] == np.array([[1.0, 2.0, 3.0]]))


def test_read_target_cols_br():
    camels = Camels()
    camels.region = "BR"
    camels.read_br_gage_flow = MagicMock(return_value=np.array([1, 2, 3]))

    gage_id_lst = ["12345678"]
    t_range = ["1990-01-01", "2000-01-01"]
    target_cols = ["streamflow_m3s"]

    result = camels.read_target_cols(gage_id_lst, t_range, target_cols)
    assert result.shape == (1, 3653, 1)
    assert np.all(result == np.array([[[1], [2], [3]]]))


def test_read_target_cols_cl():
    camels = Camels()
    camels.region = "CL"
    camels.data_source_description = {
        "CAMELS_FLOW_DIR": ["/path/to/flow_dir1", "/path/to/flow_dir2"]
    }

    gage_id_lst = ["12345678"]
    t_range = ["1990-01-01", "2000-01-01"]
    target_cols = ["streamflow_m3s"]

    with patch("pandas.read_csv") as mock_read_csv:
        mock_df = pd.DataFrame(
            {"1990-01-01": [1.0], "1990-01-02": [2.0], "1990-01-03": [3.0]}
        ).T
        mock_df.index = pd.to_datetime(mock_df.index)
        mock_read_csv.return_value = mock_df

        result = camels.read_target_cols(gage_id_lst, t_range, target_cols)
        assert result.shape == (1, 3653, 1)
        assert np.all(result[:, :3, 0] == np.array([[1.0, 2.0, 3.0]]))


def test_read_target_cols_gb():
    camels = Camels()
    camels.region = "GB"
    camels.read_gb_gage_flow_forcing = MagicMock(return_value=np.array([1, 2, 3]))

    gage_id_lst = ["12345678"]
    t_range = ["1990-01-01", "2000-01-01"]
    target_cols = ["discharge_spec"]

    result = camels.read_target_cols(gage_id_lst, t_range, target_cols)
    assert result.shape == (1, 3653, 1)
    assert np.all(result == np.array([[[1], [2], [3]]]))


def test_read_augmented_camels_streamflow_before_2015():
    camels = Camels()
    camels.read_usgs_gage = MagicMock(return_value=np.array([1, 2, 3]))
    camels.read_camels_streamflow = MagicMock(return_value=np.array([4, 5, 6]))

    gage_id_lst = ["01013500"]
    t_range = ["1990-01-01", "2000-01-01"]
    t_range_list = np.array(
        ["1990-01-01", "1995-01-01", "2000-01-01"], dtype="datetime64[D]"
    )

    result = camels._read_augmented_camels_streamflow(
        gage_id_lst, t_range, t_range_list, 0
    )
    assert np.array_equal(result, np.array([1, 2, 3]))


def test_read_augmented_camels_streamflow_after_2015():
    camels = Camels()
    camels.read_usgs_gage = MagicMock(return_value=np.array([1, 2, 3]))
    camels.read_camels_streamflow = MagicMock(return_value=np.array([4, 5, 6]))

    gage_id_lst = ["01013500"]
    t_range = ["2016-01-01", "2020-01-01"]
    t_range_list = np.array(
        ["2016-01-01", "2017-01-01", "2020-01-01"], dtype="datetime64[D]"
    )

    result = camels._read_augmented_camels_streamflow(
        gage_id_lst, t_range, t_range_list, 0
    )
    assert np.array_equal(result, np.array([4, 5, 6]))


def test_read_augmented_camels_streamflow_cross_2015():
    camels = Camels()
    camels.read_usgs_gage = MagicMock(return_value=np.array([1, 2, 3]))
    camels.read_camels_streamflow = MagicMock(return_value=np.array([4, 5, 6]))

    gage_id_lst = ["01013500"]
    t_range = ["2014-01-01", "2016-01-01"]
    t_range_list = np.array(
        ["2014-01-01", "2015-01-01", "2016-01-01"], dtype="datetime64[D]"
    )

    result = camels._read_augmented_camels_streamflow(
        gage_id_lst, t_range, t_range_list, 0
    )
    assert np.array_equal(result, np.array([1, 2, 3, 4, 5, 6]))


def test_read_camels_us_model_output_data():
    camels = Camels()
    camels.data_source_dir = "/path/to/data_source"
    camels.sites = pd.DataFrame({"gauge_id": ["01013500"], "huc_02": ["01"]})

    gage_id_lst = ["01013500"]
    t_range = ["1990-01-01", "2000-01-01"]
    var_lst = ["SWE", "PRCP"]

    with patch("pandas.read_csv") as mock_read_csv:
        mock_df = pd.DataFrame(
            {
                "YR": [1990, 1990, 1990],
                "MNTH": [1, 1, 1],
                "DY": [1, 2, 3],
                "SWE": [1.0, 2.0, 3.0],
                "PRCP": [0.1, 0.2, 0.3],
            }
        )
        mock_read_csv.return_value = mock_df

        result = camels.read_camels_us_model_output_data(gage_id_lst, t_range, var_lst)
        assert result.shape == (1, 3652, 2)
        assert np.all(result[:, :3, 0] == np.array([[1.0, 2.0, 3.0]]))


def test_read_camels_us_model_output_data_invalid_var():
    camels = Camels()
    gage_id_lst = ["01013500"]
    t_range = ["1990-01-01", "2000-01-01"]
    var_lst = ["INVALID_VAR"]

    try:
        camels.read_camels_us_model_output_data(gage_id_lst, t_range, var_lst)
    except RuntimeError as e:
        assert str(e) == "not in this list"


def test_read_camels_us_model_output_data_no_data():
    camels = Camels()
    camels.data_source_dir = "/path/to/data_source"
    camels.sites = pd.DataFrame({"gauge_id": ["01013500"], "huc_02": ["01"]})

    gage_id_lst = ["01013500"]
    t_range = ["1990-01-01", "2000-01-01"]
    var_lst = ["SWE", "PRCP"]

    with patch("pandas.read_csv") as mock_read_csv:
        mock_read_csv.return_value = pd.DataFrame(
            columns=["YR", "MNTH", "DY", "SWE", "PRCP"]
        )

        result = camels.read_camels_us_model_output_data(gage_id_lst, t_range, var_lst)
        assert result.shape == (1, 3653, 2)
        assert np.all(np.isnan(result))


def test_read_mean_prcp_aus():
    camels = Camels()
    camels.region = "AUS"
    camels.read_constant_cols = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))

    gage_id_lst = ["12345678", "12345679", "12345680"]
    result = camels.read_mean_prcp(gage_id_lst)
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))


def test_read_mean_prcp_br():
    camels = Camels()
    camels.region = "BR"
    camels.read_constant_cols = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))

    gage_id_lst = ["12345678", "12345679", "12345680"]
    result = camels.read_mean_prcp(gage_id_lst)
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))


def test_read_mean_prcp_gb():
    camels = Camels()
    camels.region = "GB"
    camels.read_constant_cols = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))

    gage_id_lst = ["12345678", "12345679", "12345680"]
    result = camels.read_mean_prcp(gage_id_lst)
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))


def test_read_mean_prcp_cl():
    camels = Camels()
    camels.region = "CL"
    camels.read_constant_cols = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))

    gage_id_lst = ["12345678", "12345679", "12345680"]
    result = camels.read_mean_prcp(gage_id_lst)
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))


def test_read_mean_prcp_invalid_region():
    camels = Camels()
    camels.region = "INVALID"

    gage_id_lst = ["12345678", "12345679", "12345680"]
    try:
        camels.read_mean_prcp(gage_id_lst)
    except NotImplementedError as e:
        assert str(e) == CAMELS_NO_DATASET_ERROR_LOG

def test_download_data_source():
    camels = Camels(download=True)
