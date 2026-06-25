# Implementation Plan: Unified Data Resolution (hydrodataset)

**Based on spec:** `docs/specs/unified-data-resolution.md`
**Scope:** This repo (hydrodataset) only. Other packages adapt in their own repos.

---

## What Changes in hydrodataset

**Goal:** `resolve_data_path` becomes the universal entry point that ALL downstream packages can delegate to — local AND cloud — by accepting an optional pre-resolved storage config.

### Change 1: Add `_load_storage()` — layered storage config

Port the layered config loading pattern from hydromodel into hydrodataset as standard behavior:

```python
def _load_storage(project_root: Path) -> Dict[str, Any]:
    """Merge storage config from layered sources.

    Priority (last wins):
    1. ~/hydro_setting.yml          (user-level)
    2. {project_root}/.hydro_setting.yml  (project-level, overrides user)
    """
    layers = [
        Path.home() / "hydro_setting.yml",
        project_root / ".hydro_setting.yml",
    ]
    storage: Dict[str, Any] = {}
    for layer in layers:
        data = _load_yaml(layer)
        layer_storage = data.get("storage")
        if isinstance(layer_storage, dict):
            storage.update(layer_storage)
    return storage
```

### Change 2: Add optional `storage` param to `resolve_data_path`

```python
def resolve_data_path(
    dataset_id: str,
    *,
    source: str = "local",
    project_root: Optional[str] = None,
    local_root: Optional[str] = None,
    storage: Optional[Dict[str, Any]] = None,       # NEW
    registry: Optional[Dict[str, Dict[str, Any]]] = None,
    extra_registry_dicts: Optional[List[Dict[str, Dict[str, str]]]] = None,
    extra_reader_aliases: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
```

Behavior:
- **`storage` provided**: use `storage["local"]["root"]` for local, `storage["s3"]` for cloud. Skips `get_local_root()` / `get_storage_config()`.
- **`storage` not provided**: call `_load_storage(project_root)` internally — backward compatible, existing callers unchanged.
- **`local_root` + `storage` both provided**: `local_root` wins for local source (explicit override).

### Change 3: Update cloud branch

Replace `get_storage_config()` with the `storage` param:

```python
# Before:
storage_cfg = get_storage_config()
s3 = storage_cfg.get("s3")

# After:
storage_cfg = storage if storage is not None else _load_storage(project_root)
s3 = storage_cfg.get("s3")
```

### Change 4: Update local branch

```python
# Before:
root_dir = local_root if local_root is not None else get_local_root()

# After:
if local_root is not None:
    root_dir = local_root
elif storage is not None:
    root_dir = Path(storage["local"]["root"])
else:
    root_dir = get_local_root()
```

### Files touched:
- `hydrodataset/configs/data_resolver.py` (+50 lines: `_load_storage`, param, branch updates)
- `hydrodataset/configs/settings.py` (no changes needed — `get_local_root` and `get_storage_config` remain as fallback)

### What does NOT change:
- Public API signature is backward compatible (new param is optional)
- `_DEFAULT_REGISTRY` — unchanged
- `READER_ALIASES` — unchanged
- `_load_registry` 3-layer cascade — unchanged
- `_validate_relative_path` — unchanged
- `_build_cloud_uri` logic — unchanged (just uses `storage` param instead of direct `get_storage_config()`)

---

## Downstream Impact (NOT in this repo)

These changes enable the other packages to simplify — but those are separate PRs in their own repos:

| Package | What they'll do after hydrodataset is updated |
|---------|----------------------------------------------|
| hydrodatasource | Remove `str(result)` wrapper, clean `Path(local_root)` conversion |
| hydromodel | Delete `_build_cloud_uri`, `_load_yaml`, `_load_storage`, `_collect_yaml_datasets`. `resolve_data_cfgs` delegates cloud to hydrodataset |
| torchhydro | Remove `resolve_dataset_uri`. Call hydrodataset directly |

---

## Verification

```bash
# Full test suite
cd D:/Code/hydrodataset
.venv/Scripts/python.exe -m pytest tests/ -v --tb=short

# Specific resolver tests
.venv/Scripts/python.exe -m pytest tests/test_configs_resolver.py -v --tb=short

# Verify _build_cloud_uri still only in hydrodataset (sanity)
grep -r "_build_cloud_uri" hydrodataset/configs/  # should show only data_resolver.py
```

---

## Risks

| Risk | Mitigation |
|------|------------|
| Layered storage changes default behavior | `storage=None` falls back to `get_local_root()` + `get_storage_config()` — identical to current behavior |
| `_load_storage` fails when `project_root` is None | Guard: `project_root = Path(project_root) if project_root else Path.cwd()` |
| Existing tests break on new param | Param is keyword-only with default `None` — all existing call sites work unchanged |
