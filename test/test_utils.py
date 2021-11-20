import numpy as np
from hydrobench.utils.hydro_utils import utc_to_local


def test_always_passes():
    assert True


def test_tz_trans():
    utc_time_str = "2020-01-01T00:00:00"
    local_tz = "Asia/Hong_Kong"
    utc_ts = utc_to_local(utc_time_str, local_tz)
    assert utc_ts == "2020-01-01T08:00:00"


def test_tz_np_trans():
    utc_time = np.datetime64("2020-01-01T00:00:00")
    utc_ts = utc_to_local(utc_time)
    assert utc_ts == "2020-01-01T08:00:00"
