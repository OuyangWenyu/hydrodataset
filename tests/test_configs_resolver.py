"""Unit tests for hydrodataset.configs.data_resolver."""

import pytest

from hydrodataset.configs.data_resolver import (
    READER_ALIASES,
    DatasetResolutionError,
    _validate_relative_path,
)


class TestReaderAliases:
    def test_all_have_required_keys(self):
        for name, spec in READER_ALIASES.items():
            assert "module" in spec, f"{name}: missing module"
            assert "class" in spec, f"{name}: missing class"
            assert "category" in spec, f"{name}: missing category"
            assert spec["category"] == "hydrodataset", f"{name}: wrong category"

    def test_known_dataset(self):
        assert "camels_us" in READER_ALIASES
        assert READER_ALIASES["camels_us"]["class"] == "CamelsUs"

    def test_count_matches_registry(self):
        # Should have 33 entries matching datasets.yml
        assert len(READER_ALIASES) == 33


class TestValidateRelativePath:
    def test_valid_relative(self):
        _validate_relative_path("camels_us", "test")
        _validate_relative_path("public/camels_us", "test")
        _validate_relative_path(".", "test")

    def test_uri_forbidden(self):
        with pytest.raises(DatasetResolutionError, match="relative"):
            _validate_relative_path("s3://bucket/path", "test")

    def test_parent_traversal_forbidden(self):
        with pytest.raises(DatasetResolutionError):
            _validate_relative_path("../../etc", "test")

    def test_absolute_windows(self):
        with pytest.raises(DatasetResolutionError):
            _validate_relative_path("C:\\data\\path", "test")

    def test_absolute_posix(self):
        with pytest.raises(DatasetResolutionError):
            _validate_relative_path("/absolute/path", "test")

    def test_non_string_raises(self):
        with pytest.raises(DatasetResolutionError):
            _validate_relative_path(123, "test")
