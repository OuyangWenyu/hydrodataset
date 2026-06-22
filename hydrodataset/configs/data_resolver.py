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
from typing import Any, Dict, Optional

import yaml

from hydrodataset.configs.settings import (
    get_local_root,
    get_storage_config,
    load_settings,
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
    "camels_deby": {
        "module": "hydrodataset.camels_deby",
        "class": "CamelsDeby",
        "category": "hydrodataset",
    },
    "camels_dk": {
        "module": "hydrodataset.camels_dk",
        "class": "CamelsDk",
        "category": "hydrodataset",
    },
    "camels_es": {
        "module": "hydrodataset.camels_es",
        "class": "CamelsEs",
        "category": "hydrodataset",
    },
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
    "mopex": {
        "module": "hydrodataset.mopex",
        "class": "Mopex",
        "category": "hydrodataset",
    },
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
    "hype": {
        "module": "hydrodataset.hype",
        "class": "Hype",
        "category": "hydrodataset",
    },
    "simbi": {
        "module": "hydrodataset.simbi",
        "class": "simbi",
        "category": "hydrodataset",
    },
    "waterbenchiowa": {
        "module": "hydrodataset.waterbenchiowa",
        "class": "waterbenchiowa",
        "category": "hydrodataset",
    },
    "hyd_responses": {
        "module": "hydrodataset.hyd_responses",
        "class": "HydResponses",
        "category": "hydrodataset",
    },
    "jialing": {
        "module": "hydrodataset.jialingriverchina",
        "class": "jialingriverchina",
        "category": "hydrodataset",
    },
}

FORBIDDEN_PATH_PATTERNS = {"://", ".."}


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return loaded or {}


def _load_registry(
    project_root: Path,
) -> Dict[str, Dict[str, Any]]:
    """Load dataset registry from configs/datasets.yml.

    Looks in two locations:
    1. {project_root}/configs/datasets.yml (project-level, takes precedence)
    2. hydrodataset/configs/datasets.yml (package fallback)

    Parameters
    ----------
    project_root : Path
        Root of the calling project.

    Returns
    -------
    dict
        Dataset registry mapping dataset_id -> {'reader': ..., 'path': ...}

    Raises
    ------
    DatasetResolutionError
        If no registry file is found.
    """
    # Project-level config (takes precedence)
    project_registry = project_root / "configs" / "datasets.yml"
    # Package-internal fallback
    package_registry = Path(__file__).resolve().parent / "datasets.yml"

    registry_path = None
    if project_registry.exists():
        registry_path = project_registry
    elif package_registry.exists():
        registry_path = package_registry

    if registry_path is None:
        raise DatasetResolutionError(
            f"Dataset registry not found. Tried:\n"
            f"  - {project_registry}\n"
            f"  - {package_registry}\n"
            "Create configs/datasets.yml in your project."
        )

    data = _load_yaml(registry_path)
    datasets = data.get("datasets")
    if not isinstance(datasets, dict):
        raise DatasetResolutionError(
            f"Dataset registry in {registry_path} must have a 'datasets' mapping."
        )
    return datasets


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
    local_root: Optional[Path] = None,
) -> Path:
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
        Root of the hydrodataset project (for finding configs/datasets.yml).
        Defaults to current working directory.
    local_root : Path, optional
        Override for the local storage root. When provided, skips reading
        storage settings and uses this path directly. Only applies when
        source is 'local'.

    Returns
    -------
    Path
        Absolute path pointing to the dataset's data directory.

    Raises
    ------
    DatasetResolutionError
        If any resolution step fails.
    """
    if source not in {"local", "cloud"}:
        raise DatasetResolutionError(
            f"source must be 'local' or 'cloud', got '{source}'"
        )

    root = Path(project_root) if project_root else Path.cwd()
    registry = _load_registry(root)

    if dataset_id not in registry:
        known = ", ".join(sorted(registry))
        raise DatasetResolutionError(
            f"Unknown dataset id '{dataset_id}'. " f"Known datasets: {known}"
        )

    dataset_spec = registry[dataset_id]
    reader = dataset_spec.get("reader")
    if not reader:
        raise DatasetResolutionError(f"Dataset '{dataset_id}' must define 'reader'")
    if reader not in READER_ALIASES:
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
                "storage.local.root is not configured. " "Set it in ~/hydro_setting.yml"
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
        return resolved

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
