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
    of `resolve_data_path` (defined in `configs/data_resolver.py`; import it
    from there, not from the top-level package).
-   `resolve_data_path` now supports cloud (S3) resolution via
    `source="cloud"` and a `ResolverContext` with `storage.s3.*` configured.
-   New `hydrodataset` CLI (`hydrodataset list|resolve|info|ids|read-ts|read-attr|config`)
    exposing resolver, dataset inspection and data reads from the command line.
-   Added native `CAMELS-PE` support via `CamelsPe`, wrapping
    `aqua_fetch.CAMELS_PE` (introduced in aqua-fetch 1.1.0).

**Improvements**:

-   `resolve_data_path` refactored into focused private helpers
    (`_load_registry`, `_lookup_relative_path`, `_resolve_local`,
    `_resolve_cloud`) for better readability and testability.
-   Reader-alias and path-safety validation moved to `_load_registry` so errors
    surface at load time rather than on the first path resolution call.
-   Upgraded `aqua-fetch` dependency to `>=1.1.0`.
-   Fixed streamflow unit metadata from `mm^3/s` to `m^3/s` and corrected
    temperature-unit mojibake (`掳C` → `°C`) across dataset wrappers.

## v0.2.5 (dev)

Unreleased; see the entries above for changes accumulated since the last
published release.
