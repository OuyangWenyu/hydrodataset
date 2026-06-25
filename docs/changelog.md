# Changelog

## Unreleased

**Breaking Changes**:

-   `resolve_data_path` no longer accepts flat keyword arguments (`project_root`,
    `local_root`, `storage`, `registry`, `extra_registry_dicts`,
    `extra_reader_aliases`).  Pass a `ResolverContext` object to the `ctx`
    parameter instead.
-   The `hydro_setting.yml` config key `local_data_path` has been replaced by
    `storage.local.root` (and optionally `storage.s3.*` for cloud access).

**New Features**:

-   `ResolverContext` dataclass — consolidates all resolver configuration
    (project root, storage, registry, extra dicts, extra aliases) into a single
    typed object.  Exported directly from the top-level `hydrodataset` package.
-   `Source = Literal["local", "cloud"]` type alias for the `source` parameter
    of `resolve_data_path`.
-   `resolve_data_path` now supports cloud (S3) resolution via
    `source="cloud"` and a `ResolverContext` with `storage.s3.*` configured.

**Improvements**:

-   `resolve_data_path` refactored into focused private helpers
    (`_load_registry`, `_lookup_relative_path`, `_resolve_local`,
    `_resolve_cloud`) for better readability and testability.
-   Reader-alias and path-safety validation moved to `_load_registry` so errors
    surface at load time rather than on the first path resolution call.

## v0.0.1 - Date

**Improvement**:

-   TBD

**New Features**:

-   TBD
