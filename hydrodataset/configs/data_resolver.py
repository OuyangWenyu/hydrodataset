"""
Lightweight data path resolver for hydrodataset.

Mirrors hydromodel's configs/data_resolver.py contract so both
repositories produce the same resolved URIs for the same inputs.

Usage:
    from hydrodataset.configs.data_resolver import resolve_data_path

    uri = resolve_data_path("camels_us")
    dataset = CamelsUs(uri)
"""

from __future__ import annotations

import dataclasses
import importlib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Literal, Optional

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
    # CAMELS Series (18)
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
    "camels_pe": {
        "module": "hydrodataset.camels_pe",
        "class": "CamelsPe",
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


def _effective_aliases(
    extra: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, Dict[str, str]]:
    """Return READER_ALIASES merged with any caller-injected extras.

    Args:
        extra: Additional reader aliases to merge on top of the built-in
            READER_ALIASES.  Extra entries take precedence on key collision.

    Returns:
        Merged dict of all known reader aliases.
    """
    aliases: Dict[str, Dict[str, str]] = dict(READER_ALIASES)
    if extra:
        aliases.update(extra)
    return aliases


# Default dataset registry (27 active entries; several are commented out).
# Maps dataset_id -> {"reader": <reader_alias>, "path": <relative_path>}.
# This is the authoritative registry for hydrodataset-served datasets.
# Other packages (e.g. hydrodatasource) inject additional entries via
# resolve_data_path(extra_registry_dicts=...).
_DEFAULT_REGISTRY: Dict[str, Dict[str, str]] = {
    # CAMELS Series (16 active; camels_deby, camels_es commented out)
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
    "camels_pe": {"reader": "camels_pe", "path": "."},
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
    if not isinstance(loaded, dict):
        return {}
    return loaded


def _load_registry(
    project_root: Path,
    extra_dicts: Optional[List[Dict[str, Dict[str, str]]]] = None,
    extra_reader_aliases: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build and validate the dataset registry with three-layer override.

    Cascade order (higher overrides lower):
    1. _DEFAULT_REGISTRY (built-in Python dict, 27 active entries)
    2. extra_dicts (injected by callers, e.g. hydrodatasource)
    3. {project_root}/configs/datasets.yml (optional user project YAML
       override; not shipped, but the code path is retained for projects that
       provide one)

    Every entry is validated for required fields, a known reader alias, and a
    safe relative path immediately after the merge so that per-dataset lookup is
    a plain dict access with no repeated validation.

    Parameters
    ----------
    project_root : Path
        Root of the calling project (for user YAML lookup).
    extra_dicts : list of dict, optional
        Additional registry dicts injected between the default and user
        YAML.  Each dict maps dataset_id -> {'reader': ..., 'path': ...}.
    extra_reader_aliases : dict, optional
        Additional reader aliases to merge with READER_ALIASES before
        validating the registry entries.

    Returns
    -------
    dict
        Merged and validated dataset registry.

    Raises
    ------
    DatasetResolutionError
        If the merged registry is empty or any entry is invalid.
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
            "No datasets registered. Supply a registry via extra_registry_dicts "
            "or a project-level configs/datasets.yml."
        )

    effective_aliases = _effective_aliases(extra_reader_aliases)

    for dataset_id, spec in registry.items():
        if not isinstance(spec, dict):
            raise DatasetResolutionError(
                f"Dataset '{dataset_id}' entry must be a dict, got {type(spec)}"
            )
        reader = spec.get("reader")
        if not reader:
            raise DatasetResolutionError(f"Dataset '{dataset_id}' must define 'reader'")
        if reader not in effective_aliases:
            raise DatasetResolutionError(
                f"Unknown reader alias '{reader}' for dataset '{dataset_id}'"
            )
        path_val = spec.get("path")
        if not path_val:
            raise DatasetResolutionError(f"Dataset '{dataset_id}' must define 'path'")
        _validate_relative_path(path_val, dataset_id)

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


def _load_storage(project_root: Path) -> Dict[str, Any]:
    """Load layered storage configuration.

    Merges storage config from two sources (last wins):
    1. ``~/hydro_setting.yml`` (user-level)
    2. ``{project_root}/.hydro_setting.yml`` (project-level override)

    Parameters
    ----------
    project_root : Path
        Root of the calling project.

    Returns
    -------
    dict
        Merged storage configuration (may be empty).
    """
    from hydrodataset.configs.settings import DEFAULT_SETTING_PATH

    layers = [
        DEFAULT_SETTING_PATH,
        project_root / ".hydro_setting.yml",
    ]
    storage: Dict[str, Any] = {}
    for layer in layers:
        data = _load_yaml(layer)
        layer_storage = data.get("storage")
        if isinstance(layer_storage, dict):
            storage.update(layer_storage)
    return storage


Source = Literal["local", "cloud"]


@dataclass(frozen=True)
class ResolverContext:
    """Immutable bundle of pre-resolved resolver inputs.

    Pass an instance via the ``ctx`` parameter of ``resolve_data_path`` to
    control registry loading, storage configuration, and reader extension.

    .. note::

        ``frozen=True`` provides **shallow** immutability — field reassignment
        is blocked, but the contained ``Dict``/``List`` fields remain mutable
        in-place.  Do not mutate ``storage``, ``registry``, or ``extra_*``
        fields after construction.

    Attributes:
        project_root: Root directory used to locate
            ``configs/datasets.yml``.  Defaults to the current working
            directory at instantiation time.
        storage: Pre-resolved storage configuration dict (same shape as the
            ``storage:`` block in ``hydro_setting.yml``).  When None, storage
            is loaded from ``~/hydro_setting.yml`` at resolve time.
        registry: Pre-loaded dataset registry.  When None, the registry is
            built from ``_DEFAULT_REGISTRY`` + ``extra_registry_dicts`` +
            the project YAML at resolve time.
        extra_registry_dicts: Additional registry dicts injected on top of
            the default registry.
        extra_reader_aliases: Additional reader aliases merged with
            ``READER_ALIASES`` during registry validation.
    """

    project_root: Path = field(default_factory=Path.cwd)
    storage: Optional[Dict[str, Any]] = None
    registry: Optional[Dict[str, Dict[str, Any]]] = None
    extra_registry_dicts: Optional[List[Dict[str, Dict[str, str]]]] = None
    extra_reader_aliases: Optional[Dict[str, Dict[str, str]]] = None


def _lookup_relative_path(
    registry: Dict[str, Dict[str, Any]],
    dataset_id: str,
) -> str:
    """Return the relative path for a dataset from the registry.

    Assumes the registry was produced by ``_load_registry`` and is therefore
    already validated.  When a pre-built registry is supplied by the caller,
    a basic existence check is still performed.

    Args:
        registry: Merged (and ideally pre-validated) dataset registry.
        dataset_id: Dataset identifier to look up.

    Returns:
        Relative path string for the dataset.

    Raises:
        DatasetResolutionError: If the dataset is unknown or has no path.
    """
    if dataset_id not in registry:
        known = ", ".join(sorted(registry))
        raise DatasetResolutionError(
            f"Unknown dataset id '{dataset_id}'. Known datasets: {known}"
        )
    path_val = registry[dataset_id].get("path")
    if not path_val:
        raise DatasetResolutionError(f"Dataset '{dataset_id}' must define 'path'")
    if not isinstance(path_val, str):
        raise DatasetResolutionError(
            f"Dataset '{dataset_id}' path must be a string, got {type(path_val)}"
        )
    _validate_relative_path(path_val, dataset_id)
    return path_val


def _resolve_local(
    relative_path: str,
    storage: Optional[Dict[str, Any]],
) -> str:
    """Resolve a dataset path against the local storage root.

    Callers pass storage via ``ResolverContext.storage``, which is the
    single, consistent source this function reads from.

    Args:
        relative_path: Validated relative path from the registry.
        storage: Storage config dict.  Uses ``storage['local']['root']`` when
            present, otherwise falls back to ``get_local_root()``.

    Returns:
        Absolute local filesystem path string.

    Raises:
        DatasetResolutionError: If root is unconfigured, missing, or the
            resolved dataset path does not exist.
    """
    if storage is not None and isinstance(storage.get("local"), dict):
        root_str = storage["local"].get("root")
        root_dir: Optional[Path] = Path(root_str) if root_str else None
    else:
        root_dir = get_local_root()
    if root_dir is None:
        if storage is not None:
            raise DatasetResolutionError(
                "storage.local.root is not configured. "
                "Set 'local.root' in your ResolverContext.storage dict."
            )
        raise DatasetResolutionError(
            "storage.local.root is not configured. Set it in ~/hydro_setting.yml"
        )
    if not root_dir.exists():
        raise DatasetResolutionError(f"storage.local.root does not exist: {root_dir}")
    resolved = root_dir / relative_path
    if not resolved.exists():
        raise DatasetResolutionError(
            f"Resolved dataset path does not exist: {resolved}"
        )
    return str(resolved)


def _resolve_cloud(
    relative_path: str,
    storage: Optional[Dict[str, Any]],
) -> str:
    """Build an S3 URI for the dataset.

    Args:
        relative_path: Validated relative path from the registry.
        storage: Storage config dict; must contain ``s3.bucket``.

    Returns:
        S3 URI string in the form ``s3://bucket/[prefix/]path``.

    Raises:
        DatasetResolutionError: If S3 config is absent or incomplete.
    """
    storage_cfg = storage if storage is not None else get_storage_config()
    s3 = storage_cfg.get("s3")
    if not isinstance(s3, dict):
        raise DatasetResolutionError("storage.s3 is required for cloud source")
    bucket = s3.get("bucket")
    if not bucket:
        raise DatasetResolutionError("storage.s3.bucket is required")
    prefix = str(s3.get("prefix") or "").strip("/")
    # Normalise Windows back-slashes and strip leading slashes before joining.
    rel = relative_path.replace("\\", "/").strip("/")
    # Treat "." as empty (bucket root) — avoids s3://bucket/.
    if rel == ".":
        rel = ""
    parts = [p for p in (prefix, rel) if p]
    key = str(PurePosixPath(*parts)) if parts else ""
    return f"s3://{bucket}/{key}"


def resolve_data_path(
    dataset_id: str,
    *,
    source: Optional[Source] = None,
    ctx: Optional[ResolverContext] = None,
) -> str:
    """Resolve a dataset id to an absolute data path.

    Combines the dataset registry entry with storage configuration to produce
    a single absolute path.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier from the registry (e.g. 'camels_us').
    source : 'local' | 'cloud', optional
        Storage backend.  When None (the default), it falls back to
        ``storage.default_source`` from the settings, or 'local' if that
        key is absent.
    ctx : ResolverContext, optional
        Pre-built resolver context.  When None, defaults are used (loads
        storage from ``~/hydro_setting.yml`` and builds registry from
        ``_DEFAULT_REGISTRY``).

    Returns
    -------
    str
        Absolute path (local source) or S3 URI (cloud source).

    Raises
    ------
    DatasetResolutionError
        If any resolution step fails.
    """
    ctx = ctx or ResolverContext()
    # Resolve storage: explicit ctx.storage wins; otherwise load from layered
    # YAML files (~/hydro_setting.yml → {project_root}/.hydro_setting.yml).
    storage = (
        ctx.storage if ctx.storage is not None else _load_storage(ctx.project_root)
    )
    # When source is not given explicitly, use storage.default_source
    # (falling back to 'local').
    if source is None:
        source = storage.get("default_source", "local")
    reg = ctx.registry or _load_registry(
        ctx.project_root,
        extra_dicts=ctx.extra_registry_dicts,
        extra_reader_aliases=ctx.extra_reader_aliases,
    )
    relative_path = _lookup_relative_path(reg, dataset_id)
    if source == "local":
        return _resolve_local(relative_path, storage)
    if source == "cloud":
        return _resolve_cloud(relative_path, storage)
    raise DatasetResolutionError(f"source must be 'local' or 'cloud', got {source!r}")


def open_dataset(
    dataset_id: str,
    *,
    source: Optional[Source] = None,
    ctx: Optional[ResolverContext] = None,
    **reader_kwargs: Any,
):
    """Resolve a dataset id and return an instantiated reader object.

    Combines ``resolve_data_path`` with dynamic import of the reader class
    registered in ``READER_ALIASES``.  The reader class is looked up via the
    registry entry's ``"reader"`` field so that ``dataset_id`` and reader alias
    need not be identical (e.g. ``songliao_event`` -> reader ``floodevent``).

    Parameters
    ----------
    dataset_id : str
        Dataset identifier from the registry (e.g. ``'camels_us'``,
        ``'songliao_event'``).
    source : 'local' | 'cloud'
        Storage backend passed to ``resolve_data_path``.
    ctx : ResolverContext, optional
        Pre-built resolver context.  ``extra_reader_aliases`` in the context
        is merged with the built-in ``READER_ALIASES`` before looking up the
        reader class, so callers that inject additional aliases (e.g.
        hydrodatasource) see their own readers automatically.
    **reader_kwargs
        Extra keyword arguments forwarded verbatim to the reader constructor
        (e.g. ``time_unit=["1D"]`` for ``SelfMadeHydroDataset``).

    Returns
    -------
    object
        An instance of the reader class registered for *dataset_id*.

    Raises
    ------
    DatasetResolutionError
        If the dataset id is unknown, the reader alias is unrecognised, or
        path resolution fails.

    Examples
    --------
    >>> ds = open_dataset("camels_us")
    >>> ds = open_dataset("camels_us", source="cloud")
    # Datasets injected by other packages (e.g. hydrodatasource via
    # extra_registry_dicts) are resolved the same way:
    >>> ds = open_dataset("songliao_event", source="cloud", ctx=ResolverContext(
    ...     extra_registry_dicts=[_HDS_REGISTRY],
    ...     extra_reader_aliases=_HDS_READER_ALIASES,
    ... ))
    """
    ctx = ctx or ResolverContext()
    reg = ctx.registry or _load_registry(
        ctx.project_root,
        extra_dicts=ctx.extra_registry_dicts,
        extra_reader_aliases=ctx.extra_reader_aliases,
    )
    if dataset_id not in reg:
        known = ", ".join(sorted(reg))
        raise DatasetResolutionError(
            f"Unknown dataset id '{dataset_id}'. Known datasets: {known}"
        )
    reader_alias = reg[dataset_id].get("reader")
    aliases = _effective_aliases(ctx.extra_reader_aliases)
    if not reader_alias or reader_alias not in aliases:
        raise DatasetResolutionError(
            f"Unknown reader alias '{reader_alias}' for dataset '{dataset_id}'"
        )
    spec = aliases[reader_alias]
    try:
        module = importlib.import_module(spec["module"])
        cls = getattr(module, spec["class"])
    except (ImportError, AttributeError) as e:
        raise DatasetResolutionError(
            f"Failed to load reader '{reader_alias}' "
            f"({spec['module']}.{spec['class']}) for dataset '{dataset_id}': {e}"
        ) from e
    # Pass the already-built registry to avoid a second _load_registry call.
    resolve_ctx = (
        ctx if ctx.registry is not None else dataclasses.replace(ctx, registry=reg)
    )
    uri = resolve_data_path(dataset_id, source=source, ctx=resolve_ctx)
    return cls(uri=uri, **reader_kwargs)
