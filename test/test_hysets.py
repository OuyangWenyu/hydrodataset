"""
Author: Wenyu Ouyang
Date: 2023-07-18 11:46:12
LastEditTime: 2023-07-18 17:05:49
LastEditors: Wenyu Ouyang
Description: Test for hysets dataset reading
FilePath: \hydrodataset\test\test_hysets.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import os
import numpy as np
from hydrodataset import ROOT_DIR
from hydrodataset.hysets import Hysets


def test_read_hysets():
    hysets = Hysets(
        os.path.join(ROOT_DIR, "hysets"),
    )
    hysets_ids = hysets.read_object_ids()
    assert len(hysets_ids) == 14425

    streamflow_types = hysets.get_target_cols()
    np.testing.assert_array_equal(streamflow_types, np.array(["discharge"]))
    focing_types = hysets.get_relevant_cols()
    np.testing.assert_array_equal(focing_types, np.array(["pr", "tasmax", "tasmin"]))
    attr_types = hysets.get_constant_cols()
    np.testing.assert_array_equal(
        attr_types[:3],
        np.array(["Centroid_Lat_deg_N", "Centroid_Lon_deg_E", "Drainage_Area_km2"]),
    )

    attrs = hysets.read_constant_cols(
        hysets_ids[:5],
        var_lst=["Centroid_Lat_deg_N", "Centroid_Lon_deg_E", "Drainage_Area_km2"],
    )
    np.testing.assert_almost_equal(
        attrs,
        np.array(
            [
                [4.72580600e01, -6.85958300e01, 1.47039211e04],
                [4.72066100e01, -6.89569400e01, 1.35864350e03],
                [4.75385000e01, -6.85918000e01, 2.71200000e03],
                [4.72375000e01, -6.85827800e01, 2.24576380e03],
                [4.70913900e01, -6.77313900e01, 1.42000000e01],
            ]
        ),
    )
    forcings = hysets.read_relevant_cols(
        hysets_ids[:5], ["1990-01-01", "2009-12-31"], var_lst=["pr", "tasmax", "tasmin"]
    )
    np.testing.assert_array_equal(
        forcings.to_array().transpose("watershed", "time", "variable").shape,
        np.array([5, 7305, 3]),
    )
    flows = hysets.read_target_cols(
        hysets_ids[:5], ["1990-01-01", "2009-12-31"], target_cols=["discharge"]
    )
    np.testing.assert_array_equal(
        flows.to_array().transpose("watershed", "time", "variable").shape,
        np.array([5, 7305, 1]),
    )
