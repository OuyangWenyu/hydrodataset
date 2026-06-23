"""Unit tests for hydrodataset.configs.data_resolver."""

import pytest

import os

from hydrodataset.configs.data_resolver import (
    READER_ALIASES,
    DatasetResolutionError,
    _DEFAULT_REGISTRY,
    _validate_relative_path,
    resolve_data_path,
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


class TestDefaultRegistry:
    """Verify _DEFAULT_REGISTRY is consistent with READER_ALIASES."""

    def test_count_matches_reader_aliases(self):
        """Every reader alias should have a corresponding registry entry."""
        assert len(_DEFAULT_REGISTRY) == len(READER_ALIASES), (
            f"_DEFAULT_REGISTRY has {len(_DEFAULT_REGISTRY)} entries, "
            f"but READER_ALIASES has {len(READER_ALIASES)}"
        )

    def test_every_entry_has_reader_and_path(self):
        for name, spec in _DEFAULT_REGISTRY.items():
            assert "reader" in spec, f"{name}: missing 'reader'"
            assert "path" in spec, f"{name}: missing 'path'"

    def test_every_reader_in_aliases(self):
        """Every reader referenced in the registry must exist in READER_ALIASES."""
        for name, spec in _DEFAULT_REGISTRY.items():
            assert spec["reader"] in READER_ALIASES, (
                f"{name}: reader '{spec['reader']}' not in READER_ALIASES"
            )


class TestExtraRegistryDicts:
    """Verify resolve_data_path handles extra_registry_dicts correctly."""

    def test_extra_dicts_override_default(self, tmp_path, monkeypatch):
        """Injected dict overrides the default registry entry for same dataset_id."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "custom_path").mkdir()

        monkeypatch.setattr(
            "hydrodataset.configs.data_resolver.get_local_root",
            lambda: data_dir,
        )

        result = resolve_data_path(
            "camels_us",
            source="local",
            project_root=str(tmp_path),
            extra_registry_dicts=[
                {"camels_us": {"reader": "camels_us", "path": "custom_path"}}
            ],
        )
        assert result == str(data_dir / "custom_path")

    def test_extra_dicts_add_new_dataset(self, tmp_path, monkeypatch):
        """Injected dict adds a new dataset not in the default registry."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "new_dataset").mkdir()

        monkeypatch.setattr(
            "hydrodataset.configs.data_resolver.get_local_root",
            lambda: data_dir,
        )

        result = resolve_data_path(
            "new_dataset",
            source="local",
            project_root=str(tmp_path),
            extra_registry_dicts=[
                {"new_dataset": {"reader": "camels_us", "path": "new_dataset"}}
            ],
        )
        assert result == str(data_dir / "new_dataset")


class TestResolveDataPathReturnType:
    """Verify resolve_data_path returns str for both local and cloud sources."""

    def test_local_returns_str(self, tmp_path, monkeypatch):
        """resolve_data_path(local) must return str, not Path."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "camels_us_data").mkdir()

        monkeypatch.setattr(
            "hydrodataset.configs.data_resolver.get_local_root",
            lambda: data_dir,
        )

        result = resolve_data_path(
            "camels_us",
            source="local",
            project_root=str(tmp_path),
            extra_registry_dicts=[
                {"camels_us": {"reader": "camels_us", "path": "camels_us_data"}}
            ],
        )
        assert isinstance(result, str), f"Expected str, got {type(result).__name__}: {result!r}"
        assert os.path.isabs(result), f"Expected absolute path, got {result!r}"

    def test_local_root_accepts_str(self, tmp_path, monkeypatch):
        """local_root must accept str (consistent with project_root and hydrodatasource)."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "camels_us_data").mkdir()

        monkeypatch.setattr(
            "hydrodataset.configs.data_resolver.get_local_root",
            lambda: data_dir,
        )

        result = resolve_data_path(
            "camels_us",
            source="local",
            local_root=str(data_dir),  # ← pass str, not Path
            extra_registry_dicts=[
                {"camels_us": {"reader": "camels_us", "path": "camels_us_data"}}
            ],
        )
        assert isinstance(result, str)
        assert result == str(data_dir / "camels_us_data")

    def test_cloud_returns_str(self, tmp_path, monkeypatch):
        """resolve_data_path(cloud) must return str (S3 URI)."""
        import hydrodataset.configs.settings as hd_settings

        # Create a temp config with S3 settings
        config_yml = tmp_path / "hydro_setting.yml"
        config_yml.write_text("""
storage:
  local:
    root: /does/not/matter
  s3:
    bucket: test-bucket
    prefix: test-prefix
""")
        monkeypatch.setattr(hd_settings, "DEFAULT_SETTING_PATH", config_yml)

        result = resolve_data_path("caravan", source="cloud")
        assert isinstance(result, str), f"Expected str, got {type(result).__name__}: {result!r}"
        assert result.startswith("s3://"), f"Expected s3:// URI, got {result!r}"
