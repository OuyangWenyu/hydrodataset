"""Unit tests for hydrodataset.configs.settings."""

import tempfile
from pathlib import Path

import pytest
import yaml

from hydrodataset.configs.settings import (
    load_settings,
    get_storage_config,
    get_local_root,
    get_default_source,
    get_cache_dir,
)


class TestLoadSettings:
    def test_nonexistent_file_returns_empty(self):
        result = load_settings(Path("/nonexistent/path/settings.yml"))
        assert result == {}

    def test_empty_file_returns_empty(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as f:
            f.write("")
            path = Path(f.name)
        try:
            result = load_settings(path)
            assert result == {}
        finally:
            path.unlink()

    def test_loads_valid_yaml(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as f:
            yaml.dump({"storage": {"local": {"root": "/tmp/test"}}}, f)
            path = Path(f.name)
        try:
            result = load_settings(path)
            assert result["storage"]["local"]["root"] == "/tmp/test"
        finally:
            path.unlink()


class TestGetStorageConfig:
    def test_new_format(self):
        settings = {"storage": {"local": {"root": "/data"}}}
        result = get_storage_config(settings)
        assert result == {"local": {"root": "/data"}}

    def test_old_format_returns_empty(self):
        settings = {"local_data_path": {"datasets-origin": "/old/data"}}
        result = get_storage_config(settings)
        assert result == {}

    def test_missing_storage_returns_empty(self):
        result = get_storage_config({})
        assert result == {}

    def test_storage_not_dict_returns_empty(self):
        result = get_storage_config({"storage": "not-a-dict"})
        assert result == {}


class TestGetLocalRoot:
    def test_returns_path(self):
        settings = {"storage": {"local": {"root": "/data/hydro"}}}
        result = get_local_root(settings)
        assert result == Path("/data/hydro")

    def test_old_format_returns_none(self):
        settings = {"local_data_path": {"datasets-origin": "/old"}}
        result = get_local_root(settings)
        assert result is None

    def test_relative_path_raises(self):
        settings = {"storage": {"local": {"root": "relative/path"}}}
        with pytest.raises(ValueError, match="must be an absolute path"):
            get_local_root(settings)


class TestGetDefaultSource:
    def test_explicit_source(self):
        settings = {"storage": {"default_source": "cloud"}}
        assert get_default_source(settings) == "cloud"

    def test_fallback_to_local(self):
        assert get_default_source({}) == "local"


class TestGetCacheDir:
    def test_absolute_cache_path(self):
        settings = {"storage": {"cache": "/tmp/my_cache"}}
        result = get_cache_dir(settings)
        assert result == Path("/tmp/my_cache")

    def test_relative_cache_with_root(self):
        settings = {
            "storage": {
                "local": {"root": "/data/hydro"},
                "cache": "cache_dir",
            }
        }
        result = get_cache_dir(settings)
        assert result == Path("/data/hydro/cache_dir")

    def test_fallback_to_home(self):
        result = get_cache_dir({})
        assert result == Path.home() / ".cache" / "hydrodataset"
