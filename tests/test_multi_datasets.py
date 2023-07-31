import os

import numpy as np
import pytest

from hydrodataset import Camels, MultiDatasets, ROOT_DIR
from hydrodataset.lamah import Lamah


@pytest.fixture()
def camels_path():
    return [
        os.path.join(ROOT_DIR, "camels", "camels_aus"),
        os.path.join(ROOT_DIR, "camels", "camels_br"),
        os.path.join(ROOT_DIR, "camels", "camels_cl"),
        os.path.join(ROOT_DIR, "camels", "camels_gb"),
        os.path.join(ROOT_DIR, "camels", "camels_us"),
        os.path.join(ROOT_DIR, "lamah_ce"),
    ]


@pytest.fixture()
def multi_datasets(camels_path):
    return MultiDatasets(
        camels_path,
        download=False,
        datasets=["CAMELS", "CAMELS", "CAMELS", "CAMELS", "CAMELS", "LamaH"],
        regions=["AUS", "BR", "CL", "GB", "US", "CE"],
    )


@pytest.fixture()
def gage_ids(multi_datasets):
    return multi_datasets.read_object_ids()


def test_which_camels_can_be_included():
    camels_series1 = MultiDatasets(
        [os.path.join(ROOT_DIR, "camels", "camels_us")],
        download=False,
        datasets=["CAMELS"],
        regions=["US"],
    )
    cs1 = camels_series1.read_object_ids()
    assert len(cs1) == 671
    camels_series2 = MultiDatasets(
        [
            os.path.join(ROOT_DIR, "camels", "camels_us"),
            os.path.join(ROOT_DIR, "lamah_ce"),
        ],
        download=False,
        datasets=["CAMELS", "LamaH"],
        regions=["US", "CE"],
    )
    cs2 = camels_series2.read_object_ids()
    assert len(cs2) == 1530


def test_get_camels_series_target_cols(multi_datasets):
    streamflow_types = multi_datasets.get_target_cols()
    np.testing.assert_array_equal(streamflow_types, np.array(["streamflow"]))


def test_get_camels_series_relevant_cols(multi_datasets):
    focing_types = multi_datasets.get_relevant_cols()
    np.testing.assert_array_equal(
        focing_types,
        np.array(["aet", "pet", "prcp", "srad", "swe", "tmax", "tmean", "tmin", "vp"]),
    )


def test_get_camels_series_constant_cols(multi_datasets):
    attr_types = multi_datasets.get_constant_cols()
    np.testing.assert_array_equal(
        attr_types,
        [
            "p_mean",
            "pet_mean",
            "aridity",
            "p_seasonality",
            "frac_snow",
            "high_prec_freq",
            "high_prec_dur",
            "high_prec_timing",
            "low_prec_freq",
            "low_prec_dur",
            "low_prec_timing",  # climate
            "elev_mean",
            "slope_mean",
            "area",  # topography
            "forest_frac",  # land cover
            "soil_depth",
            "soil_conductivity",
            "sand_frac",
            "silt_frac",
            "clay_frac",  # soil
            "geol_1st_class",
            "geol_1st_class_frac",
        ],
    )


def test_read_camels_series_constant_cols(multi_datasets, gage_ids):
    attrs = multi_datasets.read_constant_cols(
        gage_ids, constant_cols=multi_datasets.get_constant_cols().tolist()
    )
    np.testing.assert_almost_equal(attrs.shape, (3836, 22))


def test_read_camels_series_relevant_cols(multi_datasets, gage_ids):
    forcings = multi_datasets.read_relevant_cols(
        gage_ids,
        ["1990-01-01", "2010-01-01"],
        relevant_cols=[
            "aet",
            "pet",
            "prcp",
            "srad",
            "swe",
            "tmax",
            "tmean",
            "tmin",
            "vp",
        ],
    )
    np.testing.assert_array_equal(forcings.shape, np.array([3836, 7305, 9]))


def test_read_camels_series_target_cols(multi_datasets, gage_ids):
    flows = multi_datasets.read_target_cols(
        gage_ids, ["1990-01-01", "2010-01-01"], target_cols=["streamflow"]
    )
    np.testing.assert_array_equal(flows.shape, np.array([3836, 7305, 1]))


def test_read_camels_us_ce_model_time_series(multi_datasets):
    gage_id = [
        "01013500",
        "01022500",
        "01030500",
        "01031500",
        "01047000",
        "01052500",
        "01054200",
        "01055000",
        "01057000",
        "01170100",
        "1",
        "2",
    ]
    t_range_list = ["1990-01-01", "2010-01-01"]
    model_output = multi_datasets.read_relevant_cols(gage_id, t_range_list, ["pet"])
    print(model_output)


def test_read_camels_ce_data(multi_datasets):
    lamah_ce_path = os.path.join(ROOT_DIR, "lamah_ce")
    lamah_ce = Lamah(lamah_ce_path, download=False, region="CE")
    gage_id = ["2", "3", "5", "6", "7", "8", "9", "10", "11", "12"]
    t_range_list = ["1990-01-01", "2010-01-01"]
    qobs1 = lamah_ce.read_target_cols(gage_id, t_range_list, ["qobs"])
    qobs2 = multi_datasets.read_target_cols(gage_id, t_range_list, ["streamflow"])
    np.testing.assert_almost_equal(qobs1, qobs2)
    prcp1 = lamah_ce.read_relevant_cols(gage_id, t_range_list, ["prec"])
    prcp2 = multi_datasets.read_relevant_cols(gage_id, t_range_list, ["prcp"])
    np.testing.assert_almost_equal(prcp1, prcp2)


def test_read_camels_aus_data(multi_datasets):
    camels_aus_path = os.path.join(ROOT_DIR, "camels", "camels_aus")
    camels_aus = Camels(camels_aus_path, download=False, region="AUS")
    gage_id = [
        "912101A",
        "912105A",
        "915011A",
        "917107A",
        "919003A",
        "919201A",
        "G9030124",
        "A0020101",
        "G0060005",
        "401015",
    ]
    t_range_list = ["1990-01-01", "2010-01-01"]
    qobs1 = camels_aus.read_target_cols(gage_id, t_range_list, ["streamflow_MLd"])
    qobs2 = multi_datasets.read_target_cols(gage_id, t_range_list, ["streamflow"])
    np.testing.assert_almost_equal(qobs1, qobs2)
    prcp1 = camels_aus.read_relevant_cols(gage_id, t_range_list, ["precipitation_SILO"])
    prcp2 = multi_datasets.read_relevant_cols(gage_id, t_range_list, ["prcp"])
    np.testing.assert_almost_equal(prcp1, prcp2)
