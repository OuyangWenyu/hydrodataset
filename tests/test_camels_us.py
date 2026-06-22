"""
Tests for the CAMELS_US dataset (hydrodataset.camels_us.CamelsUs).

Test organization:
- **Unit tests** — test resolver, constructor, metadata. No real data needed, run in CI.
- **Integration tests** — test actual data reading. Require CAMELS_US dataset on disk,
  configured via ~/hydro_setting.yml (storage.local.root + datasets.yml).

Path resolution (ADR 0001):
  Each user configures their own storage.local.root in ~/hydro_setting.yml.
  Tests call resolve_data_path("camels_us") which reads the settings +
  datasets registry to produce an absolute URI. This means the same test
  code works across all developer machines — only the YAML config differs.
"""

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from hydrodataset import resolve_data_path, DatasetResolutionError
from hydrodataset.camels_us import CamelsUs

# ── Shared mock fixture for unit tests ──────────────────────────────
# CamelsUs.__init__ calls CAMELS_US(data_path) which triggers
# aqua_fetch initialization (file checks, potential network calls).
# Unit tests mock this to avoid external dependencies.


@pytest.fixture
def mock_aqua_camels():
    """Patch CAMELS_US from aqua_fetch so unit tests don't hit network."""
    with patch("hydrodataset.camels_us.CAMELS_US", autospec=True) as mock:
        # Configure mock with minimal attributes needed by base class methods
        mock.return_value.dynamic_features = []
        mock.return_value.static_features = []
        yield mock


# ── Path resolution (module-level, runs once at collection time) ────
# Tries to resolve the CAMELS_US data path from ~/hydro_setting.yml +
# configs/datasets.yml. If not configured, integration tests are skipped.

try:
    CAMELS_US_PATH = resolve_data_path("camels_us")
except DatasetResolutionError as _exc:
    CAMELS_US_PATH = None
    _RESOLVE_ERROR_MSG = str(_exc)

# ── Skip conditions ─────────────────────────────────────────────────

# Combined marker: skip if data not available OR running in CI
_missing = CAMELS_US_PATH is None
_ci = bool(os.getenv("CI"))
_skip_reasons = []
if _missing:
    _skip_reasons.append(
        "CAMELS_US data not available — configure storage.local.root in "
        "~/hydro_setting.yml"
    )
if _ci:
    _skip_reasons.append("not suitable for CI (large dataset download)")

needs_data = pytest.mark.skipif(
    _missing or _ci,
    reason="; ".join(_skip_reasons),
)


# ── Helper: check if data is configured (for tests that need a real path) ──

def _require_data():
    """Skip the current test if CAMELS_US data is not available."""
    if CAMELS_US_PATH is None:
        pytest.skip(_RESOLVE_ERROR_MSG)


# ═══════════════════════════════════════════════════════════════════
# Unit tests — no real data needed
# ═══════════════════════════════════════════════════════════════════


class TestResolver:
    """Tests for camels_us in the data resolver / registry."""

    def test_camels_us_in_reader_aliases(self):
        """camels_us must be a registered reader alias."""
        from hydrodataset.configs.data_resolver import READER_ALIASES

        assert "camels_us" in READER_ALIASES
        spec = READER_ALIASES["camels_us"]
        assert spec["class"] == "CamelsUs"
        assert spec["module"] == "hydrodataset.camels_us"
        assert spec["category"] == "hydrodataset"

    def test_resolve_returns_absolute_path(self):
        """resolve_data_path should return an absolute path pointing to existing directory."""
        _require_data()
        path = resolve_data_path("camels_us")
        assert os.path.isabs(path), f"Expected absolute path, got: {path}"
        assert os.path.isdir(path), f"Path does not exist: {path}"


class TestCamelsUsConstruction:
    """Tests for CamelsUs.__init__ behavior.

    All tests use mock_aqua_camels to avoid triggering aqua_fetch network calls
    during construction with an empty tmp_path.
    """

    def test_constructor_accepts_absolute_path(self, tmp_path, mock_aqua_camels):
        """Constructor must accept an absolute path and set data_source_dir."""
        ds = CamelsUs(str(tmp_path))
        assert ds.data_source_dir == tmp_path

    def test_constructor_rejects_relative_path(self, mock_aqua_camels):
        """Constructor must reject relative paths with a helpful error message."""
        with pytest.raises(ValueError, match="data_path must be an absolute path"):
            CamelsUs("relative/path/to/data")

    def test_constructor_creates_dir_if_missing(self, tmp_path, mock_aqua_camels):
        """Constructor should create data_source_dir if it does not exist."""
        new_dir = tmp_path / "new_camels_us_dir"
        assert not new_dir.exists()
        ds = CamelsUs(str(new_dir))
        try:
            assert ds.data_source_dir == new_dir
            assert new_dir.exists()
        finally:
            if new_dir.exists():
                new_dir.rmdir()

    def test_default_region_is_us(self, tmp_path, mock_aqua_camels):
        """Default region should be 'US'."""
        ds = CamelsUs(str(tmp_path))
        assert ds.region == "US"

    def test_explicit_region(self, tmp_path, mock_aqua_camels):
        """Region should be settable via the region parameter."""
        ds = CamelsUs(str(tmp_path), region="US")
        assert ds.region == "US"


class TestCamelsUsMetadata:
    """Tests for CamelsUs class-level metadata (no data needed)."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, mock_aqua_camels):
        self.ds = CamelsUs(str(tmp_path))

    def test_attributes_cache_filename(self):
        """Attribute cache filename should be set."""
        assert self.ds._attributes_cache_filename == "camels_us_attributes.nc"

    def test_timeseries_cache_filename(self):
        """Timeseries cache filename should be set."""
        assert self.ds._timeseries_cache_filename == "camels_us_timeseries.nc"

    def test_default_t_range(self):
        """Default time range should cover the CAMELS-US observation period."""
        t_range = self.ds.default_t_range
        assert t_range == ["1980-01-01", "2014-12-31"]

    def test_static_definitions_has_expected_keys(self):
        """Merged static definitions should contain both base and subclass attributes."""
        defs = self.ds._static_variable_definitions
        # Base class attributes
        assert "area" in defs
        assert "p_mean" in defs
        # Subclass attributes
        assert "elev_mean" in defs
        assert "slope_mean" in defs
        assert "gauge_lat" in defs
        assert "gauge_lon" in defs

    def test_subclass_static_definitions_has_expected_keys(self):
        """Subclass-specific static definitions should contain CAMELS-US attributes."""
        defs = self.ds._subclass_static_definitions
        assert "elev_mean" in defs
        assert "slope_mean" in defs
        assert "gauge_lat" in defs
        assert "gauge_lon" in defs
        assert "huc_02" in defs  # Unique to CAMELS-US

    def test_subclass_static_definitions_structure(self):
        """Each static definition must have specific_name and unit."""
        for var_name, spec in self.ds._subclass_static_definitions.items():
            assert "specific_name" in spec, f"{var_name}: missing specific_name"
            assert "unit" in spec, f"{var_name}: missing unit"

    def test_dynamic_variable_mapping_has_expected_keys(self):
        """Dynamic variable mapping should contain standard meteorological variables."""
        from hydrodataset import StandardVariable

        mapping = self.ds._dynamic_variable_mapping
        required = [
            StandardVariable.STREAMFLOW,
            StandardVariable.PRECIPITATION,
            StandardVariable.TEMPERATURE_MAX,
            StandardVariable.TEMPERATURE_MIN,
        ]
        for v in required:
            assert v in mapping, f"Missing required dynamic variable: {v}"

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

    def test_pet_variable_in_mapping(self):
        """PET and ET should be available as dynamic variables."""
        from hydrodataset import StandardVariable

        mapping = self.ds._dynamic_variable_mapping
        assert StandardVariable.POTENTIAL_EVAPOTRANSPIRATION in mapping
        assert StandardVariable.EVAPOTRANSPIRATION in mapping


class TestCamelsUsDynamicFeatures:
    """Tests for _dynamic_features (PET override)."""

    def test_dynamic_features_includes_pet_and_et(self, tmp_path, mock_aqua_camels):
        """_dynamic_features should include PET and ET from the override."""
        ds = CamelsUs(str(tmp_path))
        features = ds._dynamic_features()
        assert "PET" in features
        assert "ET" in features


# ═══════════════════════════════════════════════════════════════════
# Integration tests — require CAMELS_US dataset on disk
# ═══════════════════════════════════════════════════════════════════


KNOWN_GAUGE = "01013500"  # St. John River at Ninemile Bridge, Maine
# Known values from CAMELS-US v1.2 camels_topo.txt:
# 01013500,46.69917,-69.71833,339.14,69.38,687.52
KNOWN_AREA_KM2 = 687.52
KNOWN_ELEV_MEAN_M = 339.14


@needs_data
class TestCamelsUsReadObjectIds:
    """Tests for read_object_ids() — the simplest integration smoke test."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = CamelsUs(CAMELS_US_PATH)

    def test_read_object_ids_returns_list(self):
        """Should return a non-empty list of gauge ID strings."""
        ids = self.ds.read_object_ids()
        assert isinstance(ids, (list, np.ndarray))
        assert len(ids) > 0
        assert all(isinstance(gid, str) for gid in ids[:5])

    def test_read_object_ids_contains_known_gauge(self):
        """Should contain the well-known gauge '01013500'."""
        ids = self.ds.read_object_ids()
        ids_list = [str(g) for g in ids]
        assert "01013500" in ids_list


@needs_data
class TestCamelsUsReadAttr:
    """Integration tests for read_attr_xrdataset with real data."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = CamelsUs(CAMELS_US_PATH)

    def test_read_area_for_known_gauge(self):
        """read_attr_xrdataset should return the known area for gauge 01013500."""
        result = self.ds.read_attr_xrdataset(
            gage_id_lst=KNOWN_GAUGE, var_lst=["area"]
        )
        area = result["area"].values
        assert len(area) == 1
        assert np.isclose(area[0], KNOWN_AREA_KM2, rtol=0.01), (
            f"Expected area={KNOWN_AREA_KM2}, got {area[0]}. "
            f"If data changed, delete cache and regenerate."
        )

    def test_read_elev_mean_for_known_gauge(self):
        """read_attr_xrdataset should return the known elevation for gauge 01013500."""
        result = self.ds.read_attr_xrdataset(
            gage_id_lst=KNOWN_GAUGE, var_lst=["elev_mean"]
        )
        elev = result["elev_mean"].values
        assert len(elev) == 1
        assert np.isclose(elev[0], KNOWN_ELEV_MEAN_M, rtol=0.01), (
            f"Expected elev_mean={KNOWN_ELEV_MEAN_M}, got {elev[0]}"
        )

    def test_read_multiple_vars(self):
        """Should be able to read multiple attributes at once."""
        result = self.ds.read_attr_xrdataset(
            gage_id_lst=[KNOWN_GAUGE], var_lst=["area", "elev_mean", "p_mean"]
        )
        for var in ["area", "elev_mean", "p_mean"]:
            assert var in result.data_vars, f"Missing variable: {var}"

    def test_read_multiple_gauges(self):
        """Should be able to read attributes for multiple gauges at once."""
        gauges = [KNOWN_GAUGE, "01022500"]
        result = self.ds.read_attr_xrdataset(
            gage_id_lst=gauges, var_lst=["area"]
        )
        assert "area" in result.data_vars
        assert len(result.basin) == 2
        area_values = result["area"].values
        assert area_values[0] > 0
        assert area_values[1] > 0

    def test_attributes_are_not_nan(self):
        """Verify key attributes are not NaN for the known gauge."""
        attrs_to_check = [
            "area", "elev_mean", "slope_mean", "p_mean",
            "gauge_lat", "gauge_lon",
        ]
        result = self.ds.read_attr_xrdataset(
            gage_id_lst=KNOWN_GAUGE, var_lst=attrs_to_check
        )
        for attr in attrs_to_check:
            assert attr in result.data_vars, f"Missing: {attr}"
            val = result[attr].values[0]
            assert not np.isnan(val), f"{attr} is NaN for {KNOWN_GAUGE}"


@needs_data
class TestCamelsUsReadTs:
    """Integration tests for read_ts_xrdataset with real data."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = CamelsUs(CAMELS_US_PATH)

    def test_read_streamflow_single_day(self):
        """Should read observed streamflow for a known gauge and single day."""
        ts_data = self.ds.read_ts_xrdataset(
            gage_id_lst=[KNOWN_GAUGE],
            var_lst=["q_cms_obs"],
            t_range=["1981-01-04", "1981-01-04"],
        )
        assert "q_cms_obs" in ts_data.data_vars
        values = ts_data["q_cms_obs"].values
        assert values.shape == (1, 1)
        val = values[0, 0]
        assert val >= 0, f"Streamflow should be non-negative, got {val}"
        assert val < 10000, f"Streamflow unreasonably large: {val}"

    def test_read_precipitation_single_day(self):
        """Should read precipitation for a known gauge and single day."""
        ts_data = self.ds.read_ts_xrdataset(
            gage_id_lst=[KNOWN_GAUGE],
            var_lst=["pcp_mm"],
            t_range=["1981-01-04", "1981-01-04"],
        )
        assert "pcp_mm" in ts_data.data_vars
        values = ts_data["pcp_mm"].values
        assert values.shape == (1, 1)
        val = values[0, 0]
        assert val >= 0, f"Precipitation should be non-negative, got {val}"

    def test_read_multiple_variables(self):
        """Should read multiple time-series variables at once."""
        ts_data = self.ds.read_ts_xrdataset(
            gage_id_lst=[KNOWN_GAUGE],
            var_lst=["q_cms_obs", "pcp_mm", "airtemp_c_max", "airtemp_c_min"],
            t_range=["1981-01-01", "1981-01-03"],
        )
        for var in ["q_cms_obs", "pcp_mm", "airtemp_c_max", "airtemp_c_min"]:
            assert var in ts_data.data_vars, f"Missing: {var}"
            assert ts_data[var].values.shape == (1, 3)

    def test_read_streamflow_multi_day(self):
        """Should return correct number of time steps for a multi-day range."""
        ts_data = self.ds.read_ts_xrdataset(
            gage_id_lst=[KNOWN_GAUGE],
            var_lst=["q_cms_obs"],
            t_range=["1981-01-01", "1981-01-10"],
        )
        assert ts_data["q_cms_obs"].values.shape == (1, 10)

    def test_read_default_t_range(self):
        """Should work with the default_t_range (long time series)."""
        ts_data = self.ds.read_ts_xrdataset(
            gage_id_lst=[KNOWN_GAUGE],
            var_lst=["q_cms_obs"],
        )
        n_days = ts_data["q_cms_obs"].values.shape[1]
        assert n_days > 10000, f"Expected >10000 days, got {n_days}"


@needs_data
class TestCamelsUsModelOutput:
    """Integration tests for read_camels_us_model_output_data (PET/ET)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = CamelsUs(CAMELS_US_PATH)

    def test_read_pet_single_day(self):
        """Should read model-output PET for a known gauge."""
        data = self.ds.read_camels_us_model_output_data(
            gage_id_lst=[KNOWN_GAUGE],
            t_range=["1981-01-04", "1981-01-04"],
            var_lst=["PET"],
        )
        assert isinstance(data, np.ndarray)
        assert data.shape == (1, 1, 1)
        val = data[0, 0, 0]
        assert val >= 0 or np.isnan(val), (
            f"PET should be non-negative (or NaN if missing), got {val}"
        )

    def test_read_pet_et_together(self):
        """Should read both PET and ET in one call."""
        data = self.ds.read_camels_us_model_output_data(
            gage_id_lst=[KNOWN_GAUGE],
            t_range=["1981-01-04", "1981-01-04"],
            var_lst=["PET", "ET"],
        )
        assert data.shape == (1, 1, 2)

    def test_read_pet_multi_day(self):
        """Should return correct shape for a multi-day range."""
        data = self.ds.read_camels_us_model_output_data(
            gage_id_lst=[KNOWN_GAUGE],
            t_range=["1981-01-01", "1981-01-10"],
            var_lst=["PET"],
        )
        assert data.shape[1] == 10


@needs_data
class TestCamelsUsCache:
    """Tests that verify cache files are created and readable."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = CamelsUs(CAMELS_US_PATH)

    def test_attributes_cache_exists(self):
        """Attribute cache file should exist after first read."""
        self.ds.read_attr_xrdataset(gage_id_lst=KNOWN_GAUGE, var_lst=["area"])
        cache_file = self.ds.cache_dir.joinpath(
            self.ds._attributes_cache_filename
        )
        assert cache_file.exists(), (
            f"Attribute cache not found at: {cache_file}"
        )

    def test_timeseries_cache_exists(self):
        """Timeseries cache file should exist after first read."""
        self.ds.read_ts_xrdataset(
            gage_id_lst=[KNOWN_GAUGE],
            var_lst=["q_cms_obs"],
            t_range=["1981-01-01", "1981-01-01"],
        )
        cache_file = self.ds.cache_dir.joinpath(
            self.ds._timeseries_cache_filename
        )
        assert cache_file.exists(), (
            f"Timeseries cache not found at: {cache_file}"
        )
