"""Unit tests for hydrodataset.configs.data_resolver."""

import os
from pathlib import Path

import pytest

from hydrodataset.configs.data_resolver import (
    READER_ALIASES,
    DatasetResolutionError,
    ResolverContext,
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
            assert (
                spec["reader"] in READER_ALIASES
            ), f"{name}: reader '{spec['reader']}' not in READER_ALIASES"


class TestExtraRegistryDicts:
    """Verify resolve_data_path handles extra_registry_dicts via ResolverContext."""

    def test_extra_dicts_override_default(self, tmp_path, monkeypatch):
        """Injected dict overrides the default registry entry for same dataset_id."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "custom_path").mkdir()

        ctx = ResolverContext(
            project_root=tmp_path,
            storage={"local": {"root": str(data_dir)}},
            extra_registry_dicts=[
                {"camels_us": {"reader": "camels_us", "path": "custom_path"}}
            ],
        )
        result = resolve_data_path("camels_us", source="local", ctx=ctx)
        assert result == str(data_dir / "custom_path")

    def test_extra_dicts_add_new_dataset(self, tmp_path, monkeypatch):
        """Injected dict adds a new dataset not in the default registry."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "new_dataset").mkdir()

        ctx = ResolverContext(
            project_root=tmp_path,
            storage={"local": {"root": str(data_dir)}},
            extra_registry_dicts=[
                {"new_dataset": {"reader": "camels_us", "path": "new_dataset"}}
            ],
        )
        result = resolve_data_path("new_dataset", source="local", ctx=ctx)
        assert result == str(data_dir / "new_dataset")


class TestResolveDataPathReturnType:
    """Verify resolve_data_path returns str for both local and cloud sources."""

    def test_local_returns_str(self, tmp_path):
        """resolve_data_path(local) must return str, not Path."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "camels_us_data").mkdir()

        ctx = ResolverContext(
            storage={"local": {"root": str(data_dir)}},
            extra_registry_dicts=[
                {"camels_us": {"reader": "camels_us", "path": "camels_us_data"}}
            ],
        )
        result = resolve_data_path("camels_us", source="local", ctx=ctx)
        assert isinstance(
            result, str
        ), f"Expected str, got {type(result).__name__}: {result!r}"
        assert os.path.isabs(result), f"Expected absolute path, got {result!r}"

    def test_cloud_returns_str(self, tmp_path, monkeypatch):
        """resolve_data_path(cloud) must return str (S3 URI)."""
        import hydrodataset.configs.settings as hd_settings

        # Create a temp config with S3 settings
        config_yml = tmp_path / "hydro_setting.yml"
        config_yml.write_text(
            """
storage:
  local:
    root: /does/not/matter
  s3:
    bucket: test-bucket
    prefix: test-prefix
"""
        )
        monkeypatch.setattr(hd_settings, "DEFAULT_SETTING_PATH", config_yml)

        result = resolve_data_path("caravan", source="cloud")
        assert isinstance(
            result, str
        ), f"Expected str, got {type(result).__name__}: {result!r}"
        assert result.startswith("s3://"), f"Expected s3:// URI, got {result!r}"


class TestLoadStorage:
    """Tests for _load_storage() — layered storage config loading."""

    def test_empty_when_no_files(self, tmp_path, monkeypatch):
        """Returns empty dict when neither YAML exists."""
        from hydrodataset.configs.data_resolver import _load_storage

        # Point DEFAULT_SETTING_PATH to non-existent file
        monkeypatch.setattr(
            "hydrodataset.configs.settings.DEFAULT_SETTING_PATH",
            tmp_path / "nonexistent.yml",
        )

        result = _load_storage(tmp_path)
        assert result == {}, f"Expected empty dict, got {result}"

    def test_project_overrides_user(self, tmp_path, monkeypatch):
        """Project .hydro_setting.yml values override user ~/hydro_setting.yml."""
        from hydrodataset.configs.data_resolver import _load_storage

        # User-level config
        user_yml = tmp_path / "user_hydro_setting.yml"
        user_yml.write_text(
            """
storage:
  local:
    root: /user/data
  s3:
    bucket: user-bucket
"""
        )
        monkeypatch.setattr(
            "hydrodataset.configs.data_resolver.Path.home",
            lambda: tmp_path,
        )
        # Make ~/hydro_setting.yml resolve to our temp user config
        monkeypatch.setattr(
            "hydrodataset.configs.settings.DEFAULT_SETTING_PATH",
            user_yml,
        )

        # Project-level config (overrides user)
        project_yml = tmp_path / ".hydro_setting.yml"
        project_yml.write_text(
            """
storage:
  local:
    root: /project/data
"""
        )

        result = _load_storage(tmp_path)
        assert (
            result["local"]["root"] == "/project/data"
        ), f"Project should override user, got {result}"
        # s3 should still come from user (not overridden by project)
        assert (
            result["s3"]["bucket"] == "user-bucket"
        ), f"User s3 should be preserved, got {result}"

    def test_reads_local_root(self, tmp_path, monkeypatch):
        """Merged storage contains local.root from user config."""
        from hydrodataset.configs.data_resolver import _load_storage

        user_yml = tmp_path / "hydro_setting.yml"
        user_yml.write_text(
            """
storage:
  local:
    root: /data/hydro
  s3:
    bucket: my-bucket
    prefix: datasets
"""
        )
        monkeypatch.setattr(
            "hydrodataset.configs.settings.DEFAULT_SETTING_PATH",
            user_yml,
        )

        result = _load_storage(tmp_path)
        assert result["local"]["root"] == "/data/hydro"
        assert result["s3"]["bucket"] == "my-bucket"
        assert result["s3"]["prefix"] == "datasets"


class TestResolveDataPathWithCtx:
    """Tests for resolve_data_path using the ResolverContext API."""

    def test_storage_in_ctx_skips_settings(self, tmp_path):
        """When storage is in ctx, get_local_root/get_storage_config are NOT called."""
        data_dir = tmp_path / "mydata"
        data_dir.mkdir()
        (data_dir / "ds").mkdir()

        ctx = ResolverContext(
            storage={"local": {"root": str(data_dir)}},
            extra_registry_dicts=[{"camels_us": {"reader": "camels_us", "path": "ds"}}],
        )
        result = resolve_data_path("camels_us", source="local", ctx=ctx)
        assert isinstance(result, str)
        assert result == str(data_dir / "ds")

    def test_storage_in_ctx_cloud(self):
        """Cloud source uses ctx.storage['s3']."""
        ctx = ResolverContext(
            storage={"s3": {"bucket": "my-bucket", "prefix": "datasets"}},
        )
        result = resolve_data_path("caravan", source="cloud", ctx=ctx)
        assert result == "s3://my-bucket/datasets/CARAVAN/Caravan/Caravan"


class TestResolverContext:
    """Tests for the ResolverContext dataclass and ctx= call convention."""

    def test_ctx_local_path(self, tmp_path):
        """ctx= API resolves a local dataset."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "ds").mkdir()

        ctx = ResolverContext(
            storage={"local": {"root": str(data_dir)}},
            extra_registry_dicts=[{"camels_us": {"reader": "camels_us", "path": "ds"}}],
        )
        result = resolve_data_path("camels_us", source="local", ctx=ctx)
        assert isinstance(result, str)
        assert result == str(data_dir / "ds")

    def test_ctx_cloud_uri(self):
        """ctx= API builds an S3 URI."""
        ctx = ResolverContext(
            storage={"s3": {"bucket": "my-bucket", "prefix": "data"}},
        )
        result = resolve_data_path("caravan", source="cloud", ctx=ctx)
        assert result == "s3://my-bucket/data/CARAVAN/Caravan/Caravan"

    def test_storage_in_ctx_takes_priority_over_settings(self, tmp_path):
        """storage set on ctx is used directly; storage.local.root in ctx is respected."""
        data_dir = tmp_path / "explicit"
        data_dir.mkdir()
        (data_dir / "ds").mkdir()

        ctx = ResolverContext(
            storage={"local": {"root": str(data_dir)}},
            extra_registry_dicts=[{"camels_us": {"reader": "camels_us", "path": "ds"}}],
        )
        result = resolve_data_path("camels_us", source="local", ctx=ctx)
        assert result == str(data_dir / "ds")

    def test_resolver_context_is_exported_from_package(self):
        """ResolverContext is importable from the top-level package."""
        from hydrodataset import ResolverContext as RC

        assert RC is ResolverContext

    def test_invalid_source_raises(self, tmp_path):
        """A non-literal source value raises DatasetResolutionError."""
        ctx = ResolverContext(
            storage={"local": {"root": str(tmp_path)}},
        )
        with pytest.raises(DatasetResolutionError, match="source"):
            resolve_data_path("camels_us", source="ftp", ctx=ctx)
