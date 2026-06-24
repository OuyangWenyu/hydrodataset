"""
Lightweight data path resolver for hydrodataset.

Mirrors hydromodel's configs/data_resolver.py contract so both
repositories produce the same resolved URIs for the same inputs.

Usage:
    from hydrodataset.configs.data_resolver import resolve_data_path

    uri = resolve_data_path("camels_us", project_root=".")
    dataset = CamelsUs(uri)
"""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath, Path
from typing import Any, Dict, List, Optional

import yaml

from hydrodataset.configs.settings import (
    get_local_root,
    get_storage_config,
)


class DatasetResolutionError(ValueError):
    """Raised when data configuration cannot be resolved deterministically."""


# Reader aliases for all hydrodataset-served datasets.
# Maps reader name -> (module, class, category).
# Category is always "hydrodataset" for datasets in this package.
# hydromodel reads this to merge into its own READER_ALIASES.
READER_ALIASES: Dict[str, Dict[str, str]] = {
    # CAMELS Series (17)
    "camels_us": {
        "module": "hydrodataset.camels_us",
        "class": "CamelsUs",
        "category": "hydrodataset",
    },
    "camels_aus": {
        "module": "hydrodataset.camels_aus",
        "class": "CamelsAus",
        "category": "hydrodataset",
    },
    "camels_br": {
        "module": "hydrodataset.camels_br",
        "class": "CamelsBr",
        "category": "hydrodataset",
    },
    "camels_ch": {
        "module": "hydrodataset.camels_ch",
        "class": "CamelsCh",
        "category": "hydrodataset",
    },
    "camels_cl": {
        "module": "hydrodataset.camels_cl",
        "class": "CamelsCl",
        "category": "hydrodataset",
    },
    "camels_col": {
        "module": "hydrodataset.camels_col",
        "class": "CamelsCol",
        "category": "hydrodataset",
    },
    "camels_de": {
        "module": "hydrodataset.camels_de",
        "class": "CamelsDe",
        "category": "hydrodataset",
    },
    # "camels_deby": {
    #     "module": "hydrodataset.camels_deby",
    #     "class": "CamelsDeby",
    #     "category": "hydrodataset",
    # },
    "camels_dk": {
        "module": "hydrodataset.camels_dk",
        "class": "CamelsDk",
        "category": "hydrodataset",
    },
    # "camels_es": {
    #     "module": "hydrodataset.camels_es",
    #     "class": "CamelsEs",
    #     "category": "hydrodataset",
    # },
    "camels_fi": {
        "module": "hydrodataset.camels_fi",
        "class": "CamelsFi",
        "category": "hydrodataset",
    },
    "camels_fr": {
        "module": "hydrodataset.camels_fr",
        "class": "CamelsFr",
        "category": "hydrodataset",
    },
    "camels_gb": {
        "module": "hydrodataset.camels_gb",
        "class": "CamelsGb",
        "category": "hydrodataset",
    },
    "camels_ind": {
        "module": "hydrodataset.camels_ind",
        "class": "CamelsInd",
        "category": "hydrodataset",
    },
    "camels_lux": {
        "module": "hydrodataset.camels_lux",
        "class": "CamelsLux",
        "category": "hydrodataset",
    },
    "camels_nz": {
        "module": "hydrodataset.camels_nz",
        "class": "CamelsNz",
        "category": "hydrodataset",
    },
    "camels_se": {
        "module": "hydrodataset.camels_se",
        "class": "CamelsSe",
        "category": "hydrodataset",
    },
    # CAMELSH Series (2)
    "camelsh": {
        "module": "hydrodataset.camelsh",
        "class": "Camelsh",
        "category": "hydrodataset",
    },
    "camelsh_kr": {
        "module": "hydrodataset.camelsh_kr",
        "class": "CamelshKr",
        "category": "hydrodataset",
    },
    # CARAVAN Series (3)
    "caravan": {
        "module": "hydrodataset.caravan",
        "class": "Caravan",
        "category": "hydrodataset",
    },
    "caravan_dk": {
        "module": "hydrodataset.caravan_dk",
        "class": "CaravanDK",
        "category": "hydrodataset",
    },
    "grdc_caravan": {
        "module": "hydrodataset.grdc_caravan",
        "class": "GrdcCaravan",
        "category": "hydrodataset",
    },
    # LamaH Series (2)
    "lamah_ce": {
        "module": "hydrodataset.lamah_ce",
        "class": "LamahCe",
        "category": "hydrodataset",
    },
    "lamah_ice": {
        "module": "hydrodataset.lamah_ice",
        "class": "LamahIce",
        "category": "hydrodataset",
    },
    # Other Public Datasets (9)
    "hysets": {
        "module": "hydrodataset.hysets",
        "class": "Hysets",
        "category": "hydrodataset",
    },
    # "mopex": {
    #     "module": "hydrodataset.mopex",
    #     "class": "Mopex",
    #     "category": "hydrodataset",
    # },
    "bull": {
        "module": "hydrodataset.bull",
        "class": "BULL",
        "category": "hydrodataset",
    },
    "estreams": {
        "module": "hydrodataset.estreams",
        "class": "Estreams",
        "category": "hydrodataset",
    },
    # "hype": {
    #     "module": "hydrodataset.hype",
    #     "class": "Hype",
    #     "category": "hydrodataset",
    # },
    "simbi": {
        "module": "hydrodataset.simbi",
        "class": "simbi",
        "category": "hydrodataset",
    },
    # "waterbenchiowa": {
    #     "module": "hydrodataset.waterbenchiowa",
    #     "class": "waterbenchiowa",
    #     "category": "hydrodataset",
    # },
    # "hyd_responses": {
    #     "module": "hydrodataset.hyd_responses",
    #     "class": "HydResponses",
    #     "category": "hydrodataset",
    # },
    # "jialing": {
    #     "module": "hydrodataset.jialingriverchina",
    #     "class": "jialingriverchina",
    #     "category": "hydrodataset",
    # },
}

FORBIDDEN_PATH_PATTERNS = {"://", ".."}

# Default dataset registry (33 entries).
# Maps dataset_id -> {"reader": <reader_alias>, "path": <relative_path>}.
# This is the authoritative registry for hydrodataset-served datasets.
# Other packages (e.g. hydrodatasource) inject additional entries via
# resolve_data_path(extra_registry_dicts=...).
# Users can override entries by placing a configs/datasets.yml in their project.
_DEFAULT_REGISTRY: Dict[str, Dict[str, str]] = {
    # CAMELS Series (17)
    "camels_us": {"reader": "camels_us", "path": "."},
    "camels_aus": {"reader": "camels_aus", "path": "."},
    "camels_br": {"reader": "camels_br", "path": "."},
    "camels_ch": {"reader": "camels_ch", "path": "."},
    "camels_cl": {"reader": "camels_cl", "path": "."},
    "camels_col": {"reader": "camels_col", "path": "."},
    "camels_de": {"reader": "camels_de", "path": "."},
    # "camels_deby": {"reader": "camels_deby", "path": "."},
    "camels_dk": {"reader": "camels_dk", "path": "."},
    # "camels_es": {"reader": "camels_es", "path": "."},
    "camels_fi": {"reader": "camels_fi", "path": "."},
    "camels_fr": {"reader": "camels_fr", "path": "."},
    "camels_gb": {"reader": "camels_gb", "path": "."},
    "camels_ind": {"reader": "camels_ind", "path": "."},
    "camels_lux": {"reader": "camels_lux", "path": "."},
    "camels_nz": {"reader": "camels_nz", "path": "."},
    "camels_se": {"reader": "camels_se", "path": "."},
    # CAMELSH Series (2)
    "camelsh": {"reader": "camelsh", "path": "."},
    "camelsh_kr": {"reader": "camelsh_kr", "path": "."},
    # CARAVAN Series (3)
    "caravan": {"reader": "caravan", "path": "CARAVAN/Caravan/Caravan"},
    "caravan_dk": {"reader": "caravan_dk", "path": "."},
    "grdc_caravan": {"reader": "grdc_caravan", "path": "."},
    # LamaH Series (2)
    "lamah_ce": {"reader": "lamah_ce", "path": "."},
    "lamah_ice": {"reader": "lamah_ice", "path": "."},
    # Other Public Datasets (9)
    "hysets": {"reader": "hysets", "path": "."},
    # "mopex": {"reader": "mopex", "path": "MOPEX"},
    "bull": {"reader": "bull", "path": "."},
    "estreams": {"reader": "estreams", "path": "."},
    # "hype": {"reader": "hype", "path": "HYPE"},
    "simbi": {"reader": "simbi", "path": "."},
    # "waterbenchiowa": {"reader": "waterbenchiowa", "path": "waterbenchiowa"},
    # "hyd_responses": {"reader": "hyd_responses", "path": "hyd_responses"},
    # "jialing": {"reader": "jialing", "path": "jialing"},
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return loaded or {}


def _load_registry(
    project_root: Path,
    extra_dicts: Optional[List[Dict[str, Dict[str, str]]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build dataset registry with three-layer override.

    Cascade order (higher overrides lower):
    1. _DEFAULT_REGISTRY (built-in Python dict, 33 entries)
    2. extra_dicts (injected by callers, e.g. hydrodatasource)
    3. {project_root}/configs/datasets.yml (user project override, YAML)

    Parameters
    ----------
    project_root : Path
        Root of the calling project (for user YAML lookup).
    extra_dicts : list of dict, optional
        Additional registry dicts injected between the default and user
        YAML.  Each dict maps dataset_id -> {'reader': ..., 'path': ...}.

    Returns
    -------
    dict
        Merged dataset registry.

    Raises
    ------
    DatasetResolutionError
        If the merged registry is empty.
    """
    registry = dict(_DEFAULT_REGISTRY)

    # Layer 2: caller-injected registry dicts
    if extra_dicts:
        for d in extra_dicts:
            registry.update(d)

    # Layer 3: user project YAML (highest priority)
    project_yml = project_root / "configs" / "datasets.yml"
    if project_yml.exists():
        data = _load_yaml(project_yml)
        if data:
            registry.update(data.get("datasets", {}))

    if not registry:
        raise DatasetResolutionError(
            "No datasets registered. Create configs/datasets.yml in your project."
        )
    return registry


def _validate_relative_path(path_value: str, dataset_id: str) -> None:
    """Ensure path is a safe relative path."""
    if not isinstance(path_value, str):
        raise DatasetResolutionError(
            f"Dataset '{dataset_id}' path must be a string, got {type(path_value)}"
        )
    for forbidden in FORBIDDEN_PATH_PATTERNS:
        if forbidden in path_value:
            raise DatasetResolutionError(
                f"Dataset '{dataset_id}' path must be relative, "
                f"not contain '{forbidden}'"
            )
    windows_path = PureWindowsPath(path_value)
    posix_path = PurePosixPath(path_value)
    if windows_path.is_absolute() or posix_path.is_absolute():
        raise DatasetResolutionError(
            f"Dataset '{dataset_id}' path must be relative, "
            f"got absolute: '{path_value}'"
        )


def resolve_data_path(
    dataset_id: str,
    *,
    source: str = "local",
    project_root: Optional[str] = None,
    local_root: Optional[str] = None,
    registry: Optional[Dict[str, Dict[str, Any]]] = None,
    extra_registry_dicts: Optional[List[Dict[str, Dict[str, str]]]] = None,
    extra_reader_aliases: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    """Resolve a dataset id to an absolute data path.

    Combines the dataset registry entry with storage configuration
    to produce a single absolute path. Follows the same contract as
    hydromodel's resolve_data_cfgs.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier from the registry (e.g. 'camels_us').
    source : str
        Storage backend: 'local' or 'cloud'.
    project_root : str, optional
        Root of the calling project (for finding configs/datasets.yml).
        Defaults to current working directory.
    local_root : str, optional
        Override for the local storage root. When provided, skips reading
        storage settings and uses this path directly. Only applies when
        source is 'local'.
    registry : dict, optional
        Pre-loaded dataset registry. When provided, skips _load_registry().
    extra_registry_dicts : list of dict, optional
        Additional dataset registry dicts to merge on top of the default
        registry. Each dict maps dataset_id -> {'reader': ..., 'path': ...}.
        Allows other packages (e.g. hydrodatasource) to register their own
        datasets without touching hydrodataset internals.
    extra_reader_aliases : dict, optional
        Additional reader aliases to merge with READER_ALIASES during
        validation. Allows callers that have their own reader classes
        (e.g. hydrodatasource's 11 readers) to pass validation.

    Returns
    -------
    str
        Absolute path (local source) or S3 URI (cloud source) pointing
        to the dataset's data directory.

    Raises
    ------
    DatasetResolutionError
        If any resolution step fails.
    """
    if source not in {"local", "cloud"}:
        raise DatasetResolutionError(
            f"source must be 'local' or 'cloud', got '{source}'"
        )

    if local_root is not None:
        local_root = Path(local_root)

    if registry is None:
        root = Path(project_root) if project_root else Path.cwd()
        registry = _load_registry(root, extra_dicts=extra_registry_dicts)

    if dataset_id not in registry:
        known = ", ".join(sorted(registry))
        raise DatasetResolutionError(
            f"Unknown dataset id '{dataset_id}'. Known datasets: {known}"
        )

    dataset_spec = registry[dataset_id]
    reader = dataset_spec.get("reader")
    if not reader:
        raise DatasetResolutionError(f"Dataset '{dataset_id}' must define 'reader'")

    effective_aliases = dict(READER_ALIASES)
    if extra_reader_aliases:
        effective_aliases.update(extra_reader_aliases)
    if reader not in effective_aliases:
        raise DatasetResolutionError(
            f"Unknown reader alias '{reader}' for dataset '{dataset_id}'"
        )

    relative_path = dataset_spec.get("path")
    if not relative_path:
        raise DatasetResolutionError(f"Dataset '{dataset_id}' must define 'path'")
    _validate_relative_path(relative_path, dataset_id)

    if source == "local":
        root_dir = local_root if local_root is not None else get_local_root()
        if root_dir is None:
            raise DatasetResolutionError(
                "storage.local.root is not configured. Set it in ~/hydro_setting.yml"
            )
        if not root_dir.exists():
            raise DatasetResolutionError(
                f"storage.local.root does not exist: {root_dir}"
            )
        resolved = root_dir / relative_path
        if not resolved.exists():
            raise DatasetResolutionError(
                f"Resolved dataset path does not exist: {resolved}"
            )
        return str(resolved)

    # cloud source
    storage = get_storage_config()
    s3 = storage.get("s3")
    if not isinstance(s3, dict):
        raise DatasetResolutionError("storage.s3 is required for cloud source")
    bucket = s3.get("bucket")
    if not bucket:
        raise DatasetResolutionError("storage.s3.bucket is required")
    prefix = str(s3.get("prefix") or "").strip("/")
    rel = relative_path.replace("\\", "/").strip("/")
    path = f"{prefix}/{rel}" if prefix else rel
    return f"s3://{bucket}/{path}"
