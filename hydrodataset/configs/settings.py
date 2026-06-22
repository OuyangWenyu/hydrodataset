"""
Storage and cache configuration for hydrodataset.

Reads ~/hydro_setting.yml (shared with hydromodel).
Supports the unified storage.* format defined in hydromodel ADR 0001.

No import-time side effects. All functions are pure — they accept
optional cached settings to avoid repeated file I/O.

Config format:
    storage:
      default_source: local
      local:
        root: D:/data/hydrodatasets
      cache: data/cache
      s3:
        bucket: hydro-data
        prefix: hydromodel
        region: us-east-1
        profile: default
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional

import yaml

DEFAULT_SETTING_PATH = Path.home() / "hydro_setting.yml"


def load_settings(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load settings from YAML file.

    Parameters
    ----------
    path : Path, optional
        Path to settings file. Defaults to ~/hydro_setting.yml.

    Returns
    -------
    dict
        Settings dictionary. Empty dict if file does not exist.
    """
    path = Path(path) if path else DEFAULT_SETTING_PATH
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return loaded or {}


def get_storage_config(
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract the storage.* block from settings.

    Parameters
    ----------
    settings : dict, optional
        Pre-loaded settings. Loads from default path if omitted.

    Returns
    -------
    dict
        Storage configuration block (may be empty).
    """
    if settings is None:
        settings = load_settings()
    storage = settings.get("storage")
    if not isinstance(storage, dict):
        return {}
    return storage


def get_local_root(
    settings: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Extract storage.local.root as a Path.

    Parameters
    ----------
    settings : dict, optional
        Pre-loaded settings.

    Returns
    -------
    Path or None
        Absolute path to local storage root, or None if not configured.
    """
    storage = get_storage_config(settings)
    local = storage.get("local")
    if not isinstance(local, dict):
        return None
    root = local.get("root")
    if not root:
        return None
    root_path = Path(root)
    # Accept both POSIX (/data/hydro) and Windows (C:\data\hydro) absolute forms
    if not (PurePosixPath(root).is_absolute() or root_path.is_absolute()):
        raise ValueError(
            f"storage.local.root must be an absolute path. Got: {root}"
        )
    return root_path


def get_default_source(
    settings: Optional[Dict[str, Any]] = None,
) -> str:
    """Extract storage.default_source. Falls back to 'local'."""
    storage = get_storage_config(settings)
    return storage.get("default_source", "local")


def get_cache_dir(
    settings: Optional[Dict[str, Any]] = None,
) -> Path:
    """Get cache directory from settings or sensible default.

    Resolution order:
    1. storage.cache (absolute path -> use directly; relative -> resolve against local root)
    2. Fallback: ~/.cache/hydrodataset/

    Parameters
    ----------
    settings : dict, optional
        Pre-loaded settings.

    Returns
    -------
    Path
        Cache directory path.
    """
    storage = get_storage_config(settings)
    cache = storage.get("cache")
    if cache:
        cache_path = Path(cache)
        # Accept both POSIX (/data/cache) and Windows (C:\data\cache) absolute forms
        if not (PurePosixPath(cache).is_absolute() or cache_path.is_absolute()):
            root = get_local_root(settings)
            if root:
                cache_path = root / cache
            else:
                cache_path = Path.home() / ".cache" / "hydrodataset" / cache
        return cache_path
    # Fallback
    return Path.home() / ".cache" / "hydrodataset"
