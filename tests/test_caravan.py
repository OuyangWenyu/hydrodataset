"""Tests for Caravan datasets (caravan, caravan_dk, grdc_caravan).

Follows the ADR 0001 unified data path resolution pattern established
in test_camels_series.py:
- resolve_data_path() for path resolution
- Standard variable names with sources parameter for timeseries
- List gage_id_lst
- np.isclose() for float comparison
"""

import os
import glob
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydrodataset import resolve_data_path, StandardVariable, DatasetResolutionError
from hydrodataset.caravan import Caravan
from hydrodataset.caravan_dk import CaravanDK
from hydrodataset.grdc_caravan import GrdcCaravan
from hydrodataset.configs.settings import get_cache_dir

from tests._paths import (
    DATA_ROOT as data_path,
    CARAVAN_PATH,
    CARAVAN_DK_PATH,
    GRDC_CARAVAN_PATH,
    needs_caravan,
    needs_caravan_dk,
    needs_grdc_caravan,
)
from tests.conftest import skip_if_ci

# ── Resolver tests ────────────────────────────────────────────────────


def test_resolve_caravan_returns_absolute_path():
    """resolve_data_path('caravan') returns an absolute path that exists."""
    p = resolve_data_path("caravan")
    assert os.path.isabs(p), f"Expected absolute path, got {p}"
    assert os.path.exists(p), f"Path does not exist: {p}"


def test_resolve_caravan_dk_returns_absolute_path():
    """resolve_data_path('caravan_dk') returns an absolute path."""
    from hydrodataset import DatasetResolutionError

    try:
        p = resolve_data_path("caravan_dk")
        assert os.path.isabs(p), f"Expected absolute path, got {p}"
    except DatasetResolutionError:
        pytest.skip("caravan_dk data not available")


def test_resolve_grdc_caravan_returns_absolute_path():
    """resolve_data_path('grdc_caravan') returns an absolute path."""
    from hydrodataset import DatasetResolutionError

    try:
        p = resolve_data_path("grdc_caravan")
        assert os.path.isabs(p), f"Expected absolute path, got {p}"
    except DatasetResolutionError:
        pytest.skip("grdc_caravan data not available")


# ── Caravan attribute tests ───────────────────────────────────────────


# Valid gauge verified against caravan_attributes.nc cache
KNOWN_GAUGE = "camels_01022500"


@needs_caravan
@skip_if_ci
def test_caravan_read_attr_area():
    """Read area attribute using standard variable name."""
    ds = Caravan(CARAVAN_PATH, region="US")
    result = ds.read_attr_xrdataset(
        gage_id_lst=[KNOWN_GAUGE], var_lst=["area"]
    )["area"].values

    # Ground-truth from raw CSV (area is in attributes_other_camels.csv)
    csv_path = os.path.join(
        CARAVAN_PATH, "attributes", "camels", "attributes_other_camels.csv"
    )
    df = pd.read_csv(csv_path, dtype={"gauge_id": str})
    df = df.set_index("gauge_id")
    expected = df.loc[KNOWN_GAUGE, "area"]
    assert np.isclose(result[0], expected, rtol=1e-6), (
        f"area mismatch: {result[0]} vs {expected}"
    )


@needs_caravan
@skip_if_ci
def test_caravan_read_attr_p_mean():
    """Read p_mean attribute using standard variable name."""
    ds = Caravan(CARAVAN_PATH, region="US")
    result = ds.read_attr_xrdataset(
        gage_id_lst=[KNOWN_GAUGE], var_lst=["p_mean"]
    )["p_mean"].values

    csv_path = os.path.join(
        CARAVAN_PATH, "attributes", "camels", "attributes_caravan_camels.csv"
    )
    df = pd.read_csv(csv_path, dtype={"gauge_id": str})
    df = df.set_index("gauge_id")
    expected = df.loc[KNOWN_GAUGE, "p_mean"]
    assert np.isclose(result[0], expected, rtol=1e-6), (
        f"p_mean mismatch: {result[0]} vs {expected}"
    )


@needs_caravan
@skip_if_ci
def test_caravan_read_multiple_attrs():
    """Read multiple attributes at once using standard variable names."""
    ds = Caravan(CARAVAN_PATH, region="US")
    result = ds.read_attr_xrdataset(
        gage_id_lst=[KNOWN_GAUGE], var_lst=["area", "p_mean"]
    )
    assert "area" in result.data_vars
    assert "p_mean" in result.data_vars


# ── Caravan timeseries tests ──────────────────────────────────────────


def _has_timeseries_cache():
    """Check if Caravan timeseries cache files exist."""
    cache_dir = get_cache_dir()
    pattern = os.path.join(str(cache_dir), "caravan_*_timeseries_batch_*.nc")
    return len(glob.glob(pattern)) > 0


needs_ts_cache = pytest.mark.skipif(
    not _has_timeseries_cache(),
    reason="Caravan timeseries cache not built -- run ds.cache_timeseries_xrdataset() first",
)


@needs_caravan
@skip_if_ci
@needs_ts_cache
def test_caravan_read_ts_streamflow():
    """Read streamflow timeseries using standard variable name + sources."""
    ds = Caravan(CARAVAN_PATH, region="US")
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=[KNOWN_GAUGE],
        var_lst=["streamflow"],
        t_range=["1981-01-01", "1981-01-03"],
        sources={"streamflow": "each_region"},
    )
    result = ts_data["streamflow"].values
    assert result.shape[0] == 1  # one gauge
    assert result.shape[1] == 3  # three days


@needs_caravan
@skip_if_ci
@needs_ts_cache
def test_caravan_read_ts_multi_var():
    """Read multiple timeseries variables using standard names + sources."""
    ds = Caravan(CARAVAN_PATH, region="US")
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=[KNOWN_GAUGE],
        var_lst=["streamflow", "precipitation", "temperature_max", "temperature_min"],
        t_range=["1981-01-01", "1981-01-03"],
        sources={
            "streamflow": "each_region",
            "precipitation": "each_region",
            "temperature_max": "each_region",
            "temperature_min": "each_region",
        },
    )
    assert "streamflow" in ts_data.data_vars
    assert "precipitation" in ts_data.data_vars
    assert "temperature_max" in ts_data.data_vars
    assert "temperature_min" in ts_data.data_vars


# ── Caravan constructor / metadata tests ──────────────────────────────


@needs_caravan
@skip_if_ci
def test_caravan_read_object_ids():
    """read_object_ids returns a non-empty array of gauge IDs."""
    ds = Caravan(CARAVAN_PATH, region="US")
    ids = ds.read_object_ids()
    assert len(ids) > 0
    assert isinstance(ids[0], (str, np.str_))


@needs_caravan
@skip_if_ci
def test_caravan_region_us():
    """Caravan with region='US' loads camels region data."""
    ds = Caravan(CARAVAN_PATH, region="US")
    assert ds.region == "US"
    assert ds.region_data_name == "camels"
    assert len(ds.sites) > 0


# ═══════════════════════════════════════════════════════════════════════
# CaravanDK — unit tests (no data needed)
# ═══════════════════════════════════════════════════════════════════════


class TestCaravanDKResolver:
    """Resolver-level tests for caravan_dk."""

    def test_caravan_dk_in_reader_aliases(self):
        """caravan_dk must be a registered reader alias."""
        from hydrodataset.configs.data_resolver import READER_ALIASES

        assert "caravan_dk" in READER_ALIASES
        spec = READER_ALIASES["caravan_dk"]
        assert spec["class"] == "CaravanDK"
        assert spec["module"] == "hydrodataset.caravan_dk"
        assert spec["category"] == "hydrodataset"

    def test_resolve_returns_absolute_path(self):
        """resolve_data_path should return an absolute path."""
        try:
            p = resolve_data_path("caravan_dk")
            assert os.path.isabs(p), f"Expected absolute path, got {p}"
        except DatasetResolutionError:
            pytest.skip("caravan_dk data not available")


class TestCaravanDKConstruction:
    """Tests for CaravanDK.__init__ validation logic.

    NOTE: Only tests that execute BEFORE the ``type()`` dynamic class creation
    in caravan_dk.py:80 are run here. The ``type()`` call is incompatible with
    MagicMock metaclasses, so full construction tests are skipped.
    """

    def test_constructor_rejects_relative_path(self):
        """Constructor must reject relative paths (checked before aqua_fetch)."""
        with pytest.raises(ValueError, match="uri must be an absolute path"):
            CaravanDK("relative/path")


class TestCaravanDKMetadata:
    """Tests for CaravanDK class-level definitions (no instance needed)."""

    def test_static_definitions_has_expected_keys(self):
        """Subclass static definitions should contain area and p_mean."""
        defs = CaravanDK._subclass_static_definitions
        assert "area" in defs
        assert "p_mean" in defs

    def test_subclass_static_definitions_structure(self):
        """Each subclass static definition must have specific_name and unit."""
        for var_name, spec in CaravanDK._subclass_static_definitions.items():
            assert "specific_name" in spec, f"{var_name}: missing specific_name"
            assert "unit" in spec, f"{var_name}: missing unit"

    def test_dynamic_variable_mapping_has_expected_keys(self):
        """Dynamic variable mapping should contain required variables."""
        mapping = CaravanDK._dynamic_variable_mapping
        required = [
            StandardVariable.STREAMFLOW,
            StandardVariable.PRECIPITATION,
            StandardVariable.TEMPERATURE_MAX,
            StandardVariable.TEMPERATURE_MIN,
            StandardVariable.POTENTIAL_EVAPOTRANSPIRATION,
        ]
        for v in required:
            assert v in mapping, f"Missing required dynamic variable: {v}"

    def test_dynamic_variable_sources_structure(self):
        """Each dynamic variable should have default_source and sources dict."""
        for std_var, spec in CaravanDK._dynamic_variable_mapping.items():
            assert "default_source" in spec, f"{std_var}: missing default_source"
            assert "sources" in spec, f"{std_var}: missing sources"
            default_src = spec["default_source"]
            assert default_src in spec["sources"], (
                f"{std_var}: default_source '{default_src}' not in sources"
            )
            for src_name, src_spec in spec["sources"].items():
                assert "specific_name" in src_spec, (
                    f"{std_var}/{src_name}: missing specific_name"
                )
                assert "unit" in src_spec, f"{std_var}/{src_name}: missing unit"


# ═══════════════════════════════════════════════════════════════════════
# CaravanDK — integration tests (read real data)
# ═══════════════════════════════════════════════════════════════════════

# read_object_ids() returns bare ids (e.g. "100006"); raw CSVs use the
# region-prefixed form "camelsdk_100006".
_CDK_GAUGE = "100006"
_CDK_RAW_GAUGE = "camelsdk_100006"


@needs_caravan_dk
@skip_if_ci
def test_read_caravan_dk_attr_xrdataset():
    """read_attr_xrdataset values match the raw Caravan-DK attribute CSVs."""
    ds = CaravanDK(CARAVAN_DK_PATH)
    p_mean_1 = ds.read_attr_xrdataset(gage_id_lst=[_CDK_GAUGE], var_lst=["p_mean"])[
        "p_mean"
    ].values
    area_1 = ds.read_attr_xrdataset(gage_id_lst=[_CDK_GAUGE], var_lst=["area"])[
        "area"
    ].values
    # Ground-truth from raw CSVs (p_mean in caravan attrs, area in other attrs)
    attr_dir = os.path.join(data_path, "Caravan_DK", "attributes", "camelsdk")
    p_df = pd.read_csv(
        os.path.join(attr_dir, "attributes_caravan_camelsdk.csv"),
        dtype={"gauge_id": str},
    )
    a_df = pd.read_csv(
        os.path.join(attr_dir, "attributes_other_camelsdk.csv"),
        dtype={"gauge_id": str},
    )
    expected_p = p_df[p_df["gauge_id"] == _CDK_RAW_GAUGE]["p_mean"].values[0]
    expected_a = a_df[a_df["gauge_id"] == _CDK_RAW_GAUGE]["area"].values[0]
    assert np.isclose(p_mean_1, expected_p, rtol=1e-6), (
        f"p_mean mismatch: {p_mean_1} vs {expected_p}"
    )
    assert np.isclose(area_1, expected_a, rtol=1e-6), (
        f"area mismatch: {area_1} vs {expected_a}"
    )


# Uses standard variable name + sources parameter (new ADR 0001 pattern).
@needs_caravan_dk
@skip_if_ci
def test_read_caravan_dk_timeseries_xrdataset():
    """read_ts_xrdataset streamflow matches the raw Caravan-DK timeseries CSV."""
    ds = CaravanDK(CARAVAN_DK_PATH)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=[_CDK_GAUGE],
        var_lst=["streamflow"],  # standard name
        t_range=["1998-08-11", "1998-08-11"],
        sources={"streamflow": "observations"},  # observed discharge (default)
    )
    result_1 = ts_data["streamflow"].values.flatten()[0]
    # Ground-truth from raw per-gauge timeseries CSV (comma-separated, date column)
    file_path = os.path.join(
        data_path,
        "Caravan_DK",
        "timeseries",
        "csv",
        "camelsdk",
        f"{_CDK_RAW_GAUGE}.csv",
    )
    df = pd.read_csv(file_path)
    result_2 = df[df["date"] == "1998-08-11"]["streamflow"].values[0]
    assert np.isclose(
        result_1, result_2, rtol=1e-6
    ), f"Expected {result_2}, got {result_1}"


# ═══════════════════════════════════════════════════════════════════════
# GrdcCaravan fixtures and helpers
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_aqua_grdc_caravan():
    """Patch GRDCCaravan aqua_fetch class so unit tests don't hit network."""
    with patch(
        "hydrodataset.grdc_caravan.GRDCCaravan", autospec=True
    ) as mock:
        mock.return_value.dynamic_features = []
        mock.return_value.static_features = []
        yield mock


# Known gauge ID verified against raw attribute CSV files
_GRDC_KNOWN_GAUGE = "GRDC_1159100"


def _require_grdc_data():
    """Skip if GRDC-Caravan data is not available on disk."""
    if GRDC_CARAVAN_PATH is None:
        pytest.skip("grdc_caravan data not available — configure ~/hydro_setting.yml")


# ═══════════════════════════════════════════════════════════════════════
# GrdcCaravan — unit tests (no data needed)
# ═══════════════════════════════════════════════════════════════════════


class TestGrdcCaravanResolver:
    """Resolver-level tests for grdc_caravan."""

    def test_grdc_caravan_in_reader_aliases(self):
        """grdc_caravan must be a registered reader alias."""
        from hydrodataset.configs.data_resolver import READER_ALIASES

        assert "grdc_caravan" in READER_ALIASES
        spec = READER_ALIASES["grdc_caravan"]
        assert spec["class"] == "GrdcCaravan"
        assert spec["module"] == "hydrodataset.grdc_caravan"
        assert spec["category"] == "hydrodataset"

    def test_resolve_returns_absolute_path(self):
        """resolve_data_path returns an absolute path that exists."""
        _require_grdc_data()
        p = resolve_data_path("grdc_caravan")
        assert os.path.isabs(p), f"Expected absolute path, got {p}"
        assert os.path.exists(p), f"Path does not exist: {p}"


class TestGrdcCaravanConstruction:
    """Tests for GrdcCaravan.__init__ (mocked aqua_fetch)."""

    def test_constructor_accepts_absolute_path(
        self, tmp_path, mock_aqua_grdc_caravan
    ):
        """Constructor must accept an absolute path."""
        ds = GrdcCaravan(str(tmp_path))
        assert ds.data_source_dir == tmp_path

    def test_constructor_rejects_relative_path(self, mock_aqua_grdc_caravan):
        """Constructor must reject relative paths."""
        with pytest.raises(ValueError, match="uri must be an absolute path"):
            GrdcCaravan("relative/path")

    def test_constructor_creates_dir_if_missing(
        self, tmp_path, mock_aqua_grdc_caravan
    ):
        """Constructor should create data_source_dir if it does not exist."""
        new_dir = tmp_path / "new_grdc_caravan"
        assert not new_dir.exists()
        ds = GrdcCaravan(str(new_dir))
        try:
            assert ds.data_source_dir == new_dir
            assert new_dir.exists()
        finally:
            if new_dir.exists():
                new_dir.rmdir()


class TestGrdcCaravanMetadata:
    """Tests for GrdcCaravan class-level metadata (mocked aqua_fetch)."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, mock_aqua_grdc_caravan):
        self.ds = GrdcCaravan(str(tmp_path))

    def test_attributes_cache_filename(self):
        """Attribute cache filename should be set."""
        assert self.ds._attributes_cache_filename == "grdc_caravan_attributes.nc"

    def test_timeseries_cache_filename(self):
        """Timeseries cache filename should be set."""
        assert self.ds._timeseries_cache_filename == "grdc_caravan_timeseries.nc"

    def test_default_t_range(self):
        """Default time range should be set."""
        assert self.ds.default_t_range == ["1950-01-02", "2023-05-18"]

    def test_static_definitions_has_expected_keys(self):
        """Merged static definitions should contain base and subclass attrs."""
        defs = self.ds._static_variable_definitions
        assert "area" in defs
        assert "p_mean" in defs

    def test_subclass_static_definitions_structure(self):
        """Each subclass static definition must have specific_name and unit."""
        for var_name, spec in self.ds._subclass_static_definitions.items():
            assert "specific_name" in spec, f"{var_name}: missing specific_name"
            assert "unit" in spec, f"{var_name}: missing unit"

    def test_dynamic_variable_mapping_has_expected_keys(self):
        """Dynamic variable mapping should contain required variables."""
        mapping = self.ds._dynamic_variable_mapping
        required = [
            StandardVariable.STREAMFLOW,
            StandardVariable.PRECIPITATION,
            StandardVariable.TEMPERATURE_MAX,
            StandardVariable.TEMPERATURE_MIN,
            StandardVariable.TEMPERATURE_MEAN,
            StandardVariable.POTENTIAL_EVAPOTRANSPIRATION,
        ]
        for v in required:
            assert v in mapping, f"Missing required dynamic variable: {v}"

    def test_streamflow_has_mm_and_cms_sources(self):
        """Streamflow should have both 'mm' and 'cms' unit sources."""
        streamflow = self.ds._dynamic_variable_mapping[StandardVariable.STREAMFLOW]
        sources = streamflow["sources"]
        assert "mm" in sources, "Missing mm/day streamflow source"
        assert "cms" in sources, "Missing m^3/s streamflow source"
        assert sources["mm"]["specific_name"] == "q_mm_obs"
        assert sources["cms"]["specific_name"] == "q_cms_obs"

    def test_dynamic_variable_sources_structure(self):
        """Each dynamic variable should have default_source and sources dict."""
        for std_var, spec in self.ds._dynamic_variable_mapping.items():
            assert "default_source" in spec, f"{std_var}: missing default_source"
            assert "sources" in spec, f"{std_var}: missing sources"
            default_src = spec["default_source"]
            assert default_src in spec["sources"], (
                f"{std_var}: default_source '{default_src}' not in sources"
            )
            for src_name, src_spec in spec["sources"].items():
                assert "specific_name" in src_spec, (
                    f"{std_var}/{src_name}: missing specific_name"
                )
                assert "unit" in src_spec, f"{std_var}/{src_name}: missing unit"


# ═══════════════════════════════════════════════════════════════════════
# GrdcCaravan — integration tests (read raw data files)
# ═══════════════════════════════════════════════════════════════════════


@needs_grdc_caravan
@skip_if_ci
class TestGrdcCaravanRawData:
    """Integration tests using raw CSV files (bypasses aqua_fetch)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        _require_grdc_data()
        # aqua_fetch appends the class name "GRDCCaravan"; the netcdf extension
        # data is double-nested under it (GRDC_Caravan_extension_nc x2).
        root = Path(GRDC_CARAVAN_PATH)
        self.attr_dir = (
            root
            / "GRDCCaravan"
            / "GRDC_Caravan_extension_nc"
            / "GRDC_Caravan_extension_nc"
            / "attributes"
            / "grdc"
        )

    def test_attributes_caravan_csv_exists(self):
        """The caravan attributes CSV should exist."""
        path = self.attr_dir / "attributes_caravan_grdc.csv"
        assert path.exists(), f"Missing: {path}"

    def test_attributes_other_csv_exists(self):
        """The other attributes CSV should exist."""
        path = self.attr_dir / "attributes_other_grdc.csv"
        assert path.exists(), f"Missing: {path}"

    def test_attributes_hydroatlas_csv_exists(self):
        """The hydroatlas attributes CSV should exist."""
        path = self.attr_dir / "attributes_hydroatlas_grdc.csv"
        assert path.exists(), f"Missing: {path}"

    def test_known_gauge_in_csv(self):
        """Known gauge ID should be present in the attribute CSV."""
        path = self.attr_dir / "attributes_caravan_grdc.csv"
        df = pd.read_csv(path, dtype={"gauge_id": str})
        assert _GRDC_KNOWN_GAUGE in df["gauge_id"].values, (
            f"Known gauge {_GRDC_KNOWN_GAUGE} not found"
        )

    def test_read_area_from_raw_csv(self):
        """Read area for known gauge directly from raw CSV."""
        path = self.attr_dir / "attributes_other_grdc.csv"
        df = pd.read_csv(path, dtype={"gauge_id": str})
        df = df.set_index("gauge_id")
        area = df.loc[_GRDC_KNOWN_GAUGE, "area"]
        assert area > 0, f"Expected positive area, got {area}"

    def test_read_p_mean_from_raw_csv(self):
        """Read p_mean for known gauge directly from raw CSV."""
        path = self.attr_dir / "attributes_caravan_grdc.csv"
        df = pd.read_csv(path, dtype={"gauge_id": str})
        df = df.set_index("gauge_id")
        p_mean = df.loc[_GRDC_KNOWN_GAUGE, "p_mean"]
        assert p_mean > 0, f"Expected positive p_mean, got {p_mean}"


# ═══════════════════════════════════════════════════════════════════════
# GrdcCaravan — standardized read_attr / read_ts value-comparison tests
# ═══════════════════════════════════════════════════════════════════════

# Raw netcdf-extension attribute dir (aqua_fetch appends class name "GRDCCaravan";
# the extension is double-nested under it).
_GRDC_ATTR_DIR = os.path.join(
    "GRDCCaravan",
    "GRDC_Caravan_extension_nc",
    "GRDC_Caravan_extension_nc",
    "attributes",
    "grdc",
)


@needs_grdc_caravan
@skip_if_ci
def test_read_grdc_caravan_attr_xrdataset():
    """read_attr_xrdataset values match the raw Caravan-GRDC attribute CSVs."""
    ds = GrdcCaravan(GRDC_CARAVAN_PATH)
    p_mean_1 = ds.read_attr_xrdataset(
        gage_id_lst=[_GRDC_KNOWN_GAUGE], var_lst=["p_mean"]
    )["p_mean"].values
    area_1 = ds.read_attr_xrdataset(
        gage_id_lst=[_GRDC_KNOWN_GAUGE], var_lst=["area"]
    )["area"].values
    attr_dir = os.path.join(data_path, _GRDC_ATTR_DIR)
    p_df = pd.read_csv(
        os.path.join(attr_dir, "attributes_caravan_grdc.csv"),
        dtype={"gauge_id": str},
    )
    a_df = pd.read_csv(
        os.path.join(attr_dir, "attributes_other_grdc.csv"),
        dtype={"gauge_id": str},
    )
    expected_p = p_df[p_df["gauge_id"] == _GRDC_KNOWN_GAUGE]["p_mean"].values[0]
    expected_a = a_df[a_df["gauge_id"] == _GRDC_KNOWN_GAUGE]["area"].values[0]
    assert np.isclose(p_mean_1, expected_p, rtol=1e-6), (
        f"p_mean mismatch: {p_mean_1} vs {expected_p}"
    )
    assert np.isclose(area_1, expected_a, rtol=1e-6), (
        f"area mismatch: {area_1} vs {expected_a}"
    )


# Raw netcdf timeseries dir for grdc (same nesting as attrs).
_GRDC_TS_DIR = os.path.join(
    "GRDCCaravan",
    "GRDC_Caravan_extension_nc",
    "GRDC_Caravan_extension_nc",
    "timeseries",
    "netcdf",
    "grdc",
)


@needs_grdc_caravan
@skip_if_ci
def test_read_grdc_caravan_timeseries_xrdataset():
    """read_ts_xrdataset streamflow matches the raw GRDC netcdf file."""
    ds = GrdcCaravan(GRDC_CARAVAN_PATH)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=[_GRDC_KNOWN_GAUGE],
        var_lst=["streamflow"],
        t_range=["1987-01-27", "1987-01-27"],
        sources={"streamflow": "mm"},
    )
    result_1 = ts_data["streamflow"].values.flatten()[0]
    # Ground-truth from raw per-gauge NetCDF
    nc_path = os.path.join(
        data_path, _GRDC_TS_DIR, f"{_GRDC_KNOWN_GAUGE}.nc"
    )
    raw_ds = xr.open_dataset(nc_path)
    raw_vals = raw_ds["streamflow"].values.flatten()
    raw_dates = pd.to_datetime(raw_ds["date"].values)
    idx = raw_dates == pd.Timestamp("1987-01-27")
    if not idx.any():
        pytest.skip("1987-01-27 not in raw data for this station")
    result_2 = raw_vals[idx][0]
    raw_ds.close()
    assert np.isclose(
        result_1, result_2, rtol=1e-6
    ), f"Expected {result_2}, got {result_1}"
