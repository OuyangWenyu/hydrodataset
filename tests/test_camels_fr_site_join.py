"""Regression tests for the CAMELS-FR site-to-station area join.

The real CAMELS-FR archive is not required: fixtures reconstruct the
one-to-many site groups reported in GitHub issue #27.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydrodataset.camels_fr import CamelsFr, join_site_attributes


def _synthetic_site_table() -> pd.DataFrame:
    """Build a small site_general table covering issue #27 groups.

    Returns:
        Site attributes indexed by 8-character ``sit_code_h3``.
    """
    return pd.DataFrame(
        {
            "sit_area_topo": [
                940.0,  # P3922520 -> two stations
                212.0,  # P6134020 -> two stations
                np.nan,  # V1015810 -> genuine source-missing area
                np.nan,  # V1257810
                np.nan,  # V1414010
                np.nan,  # V1464310
                np.nan,  # V3424310
                55.0,  # A1402020 -> one-to-one control
            ],
            "sit_name": [
                "site_p392",
                "site_p613",
                "site_v101",
                "site_v125",
                "site_v141",
                "site_v146",
                "site_v342",
                "site_a140",
            ],
        },
        index=[
            "P3922520",
            "P6134020",
            "V1015810",
            "V1257810",
            "V1414010",
            "V1464310",
            "V3424310",
            "A1402020",
        ],
    )


def _synthetic_station_table() -> pd.DataFrame:
    """Build a small station table with colliding 8-character prefixes.

    Returns:
        Station attributes indexed by 10-character ``sta_code_h3``.
    """
    station_ids = [
        "P392252001",
        "P392252002",
        "P613402001",
        "P613402002",
        "V101581001",
        "V125781001",
        "V141401001",
        "V146431001",
        "V342431001",
        "A140202001",
    ]
    return pd.DataFrame(
        {"sta_y_w84": np.arange(len(station_ids), dtype=float)},
        index=station_ids,
    )


# Groups from GitHub issue #27 whose dict-rename join dropped the first sibling.
_ONE_TO_MANY_AREAS = {
    "P392252001": 940.0,
    "P392252002": 940.0,
    "P613402001": 212.0,
    "P613402002": 212.0,
}

_GENUINE_MISSING = [
    "V101581001",
    "V125781001",
    "V141401001",
    "V146431001",
    "V342431001",
]


def test_dict_rename_drops_first_sibling_station():
    """The old ``{stn[:-2]: stn}`` mapping overwrites duplicate prefixes."""
    static = _synthetic_station_table()
    site = _synthetic_site_table()
    buggy_map = {stn[:-2]: stn for stn in static.index}
    renamed = site.rename(index=buggy_map)
    joined = pd.concat([renamed, static], axis=1)

    assert buggy_map["P3922520"] == "P392252002"
    assert buggy_map["P6134020"] == "P613402002"
    assert pd.isna(joined.loc["P392252001", "sit_area_topo"])
    assert joined.loc["P392252002", "sit_area_topo"] == pytest.approx(940.0)
    assert pd.isna(joined.loc["P613402001", "sit_area_topo"])
    assert joined.loc["P613402002", "sit_area_topo"] == pytest.approx(212.0)


def test_join_site_attributes_keeps_all_sibling_areas():
    """Every station of a shared site receives that site's ``sit_area_topo``."""
    static = _synthetic_station_table()
    site = _synthetic_site_table()
    joined = join_site_attributes(static, site)

    for station_id, expected in _ONE_TO_MANY_AREAS.items():
        value = joined.loc[station_id, "sit_area_topo"]
        assert np.isfinite(value), f"{station_id} area should be finite, got {value}"
        assert value > 0, f"{station_id} area should be positive, got {value}"
        assert value == pytest.approx(expected)

    for station_id in _GENUINE_MISSING:
        assert pd.isna(joined.loc[station_id, "sit_area_topo"])

    assert joined.loc["A140202001", "sit_area_topo"] == pytest.approx(55.0)
    assert list(joined.index) == list(static.index)
    assert joined.loc["P392252001", "sta_y_w84"] == pytest.approx(0.0)
    assert joined.loc["P392252002", "sta_y_w84"] == pytest.approx(1.0)


def test_overlay_joined_site_areas_repairs_local_netcdf_cache(tmp_path):
    """Local NetCDF overlay writes the same one-to-many areas as the join."""
    attr_dir = tmp_path.joinpath(*CamelsFr._ATTR_REL.split("/"))
    attr_dir.mkdir(parents=True)
    site = _synthetic_site_table()
    site.index.name = "sit_code_h3"
    site.to_csv(attr_dir / "CAMELS_FR_site_general_attributes.csv", sep=";")

    basins = _synthetic_station_table().index.to_numpy()
    # Simulate the AquaFetch dict-rename cache: first sibling is NaN.
    cached_area = np.array(
        [
            np.nan,
            940.0,
            np.nan,
            212.0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            55.0,
        ]
    )
    ds_attr = xr.Dataset(
        {"area_km2": ("basin", cached_area)},
        coords={"basin": basins},
    )
    ds_attr["area_km2"].attrs["units"] = "km^2"

    reader = object.__new__(CamelsFr)
    reader.data_source_dir = tmp_path
    repaired = reader._overlay_joined_site_areas(ds_attr)

    areas = repaired["area_km2"].to_series()
    for station_id, expected in _ONE_TO_MANY_AREAS.items():
        value = areas.loc[station_id]
        assert np.isfinite(value)
        assert value > 0
        assert value == pytest.approx(expected)
    for station_id in _GENUINE_MISSING:
        assert pd.isna(areas.loc[station_id])
    assert areas.loc["A140202001"] == pytest.approx(55.0)
    assert repaired["area_km2"].attrs["units"] == "km^2"
