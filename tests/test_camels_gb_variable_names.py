"""Regression tests for CAMELS-GB cache variable names."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydrodataset.camels_gb import CamelsGb


@pytest.mark.parametrize(
    "variables",
    [
        ["temperature_mean"],
        ["relative_humidity"],
        ["temperature_mean", "relative_humidity"],
    ],
)
def test_read_standard_variables_from_cache(monkeypatch, variables):
    reader = object.__new__(CamelsGb)
    cached = xr.Dataset(
        {
            "airtemp_c_mean": (("basin", "time"), [[8.5, 9.0]]),
            "rh_": (("basin", "time"), [[5.2, 5.4]]),
        },
        coords={
            "basin": ["102001"],
            "time": pd.date_range("1981-01-04", periods=2),
        },
    )
    monkeypatch.setattr(reader, "_load_ts_dataset", lambda **kwargs: cached)

    result = reader.read_ts_xrdataset(
        gage_id_lst=["102001"],
        var_lst=variables,
        t_range=["1981-01-04", "1981-01-05"],
    )

    expected = {
        "temperature_mean": [[8.5, 9.0]],
        "relative_humidity": [[5.2, 5.4]],
    }
    assert set(result.data_vars) == set(variables)
    for variable in variables:
        np.testing.assert_allclose(result[variable].values, expected[variable])
