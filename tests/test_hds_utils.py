import os
import pytest
from hydrodataset import SETTING
from hydrodataset.camels import Camels
from hydrodataset.hds_utils import usgs_screen_streamflow


def test_usgs_screen_streamflow():
    data_dir = SETTING["local_data_path"]["datasets-origin"]
    camels_dir = os.path.join(data_dir, "camels", "camels_us")
    camels_dataset = Camels(camels_dir)
    usgs_ids = ["01013500", "01052500", "01142500"]
    time_range = ["2014-01-01", "2020-12-31"]
    result = usgs_screen_streamflow(camels_dataset, usgs_ids, time_range)
    assert len(result) == 3
