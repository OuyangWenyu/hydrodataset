# Spec: Unified Data Resolution — ONE Function, ONE Package

**Status:** DRAFT — awaiting review
**Date:** 2026-06-24
**Decision:** Variation G — hydrodataset is the single resolution engine

---

## Objective

Make `hydrodataset.configs.data_resolver.resolve_data_path()` the **one and only** function that resolves a `dataset_id` to an absolute URI, across all four hydro-* packages.

Downstream packages (hydrodatasource, hydromodel, torchhydro) become **pure data providers**: they export reader aliases and dataset registry entries as plain dicts. They do NOT contain path-building, S3-construction, YAML-loading, or any other resolution logic.

**Who this is for:** Researchers writing experiment configs. They just want `"camels_us"` → `/data/camels_us` or `"songliao_event"` → `/data/projects/songliao/event`.

**Success looks like:**
- Exactly ONE `_build_cloud_uri` exists (in hydrodataset)
- Exactly ONE `_validate_relative_path` exists (in hydrodataset)
- Exactly ONE `_load_yaml` exists (in hydrodataset)
- hydromodel's `data_resolver.py` shrinks from 352 → ~80 lines
- hydrodatasource's `data_resolver.py` shrinks from 169 → ~100 lines (pure data + re-export)
- All existing tests pass
- A researcher can `from hydrodataset import resolve_data_path; resolve_data_path("camels_us")` and get the right path

---

## Tech Stack

- Python 3.10+
- pyyaml (already a dependency)
- pathlib (stdlib)
- No new dependencies

---

## Commands

```bash
# hydrodataset
cd D:/Code/hydrodataset
.venv/Scripts/python.exe -m pytest tests/ -v --tb=short

# hydrodatasource
cd D:/Code/hydrodatasource
.venv/Scripts/python.exe -m pytest tests/ -v --tb=short

# hydromodel
cd D:/Code/hydromodel
.venv/Scripts/python.exe -m pytest tests/ -v --tb=short

# torchhydro
cd D:/Code/torchhydro
.venv/Scripts/python.exe -m pytest tests/ -v --tb=short
```

---

## Project Structure (target state)

```
hydrodataset/configs/data_resolver.py   ← THE ENGINE (~550 lines)
  ├─ READER_ALIASES (33 entries)
  ├─ _DEFAULT_REGISTRY (33 entries)
  ├─ resolve_data_path()                ← THE one function
  ├─ _load_registry()                   ← 3-layer cascade (default → extra → user YAML)
  ├─ _load_storage()                    ← layered config (user → project .hydro_setting.yml)
  ├─ _validate_relative_path()
  ├─ _load_yaml()
  └─ _build_cloud_uri()

hydrodatasource/configs/data_resolver.py ← DATA PROVIDER + CONVENIENCE (~100 lines)
  ├─ HDS_READER_ALIASES (11 entries)    ← public export
  ├─ HDS_DATASETS (1+ entries)          ← public export
  ├─ READER_ALIASES (merged 44)         ← convenience merge
  └─ resolve_data_path()                ← thin wrapper: injects HDS dicts, delegates

hydromodel/configs/data_resolver.py      ← TASK CONFIG VALIDATOR (~80 lines)
  ├─ HYDROMODEL_READER_ALIASES (1 entry: zarr_timeseries)
  ├─ READER_ALIASES (merged: HD + HDS + HM)
  ├─ resolve_data_cfgs()                ← validates + delegates to hydrodataset
  └─ resolve_config()                   ← wraps resolve_data_cfgs
  DELETED: _build_cloud_uri, _load_yaml, _load_storage, _collect_yaml_datasets

torchhydro/configs/data_resolver.py      ← CONFIG MIGRATION SHIM (~60 lines)
  ├─ normalize_source_cfgs()            ← old→new config format migration
  DELETED: resolve_dataset_uri()         ← callers use hydrodataset directly
```

---

## Code Style

Follow surrounding conventions in `hydrodataset/configs/data_resolver.py`:

```python
"""
One-line module docstring.

Extended description.
"""

from __future__ import annotations

from hydrodataset.configs.data_resolver import (
    DatasetResolutionError,
    ResolverContext,
    resolve_data_path as _hd_resolve_data_path,
)
from typing import Dict, Optional

# Module-level constants are UPPER_SNAKE_CASE
HDS_DATASETS: Dict[str, Dict[str, str]] = {
    "songliao_event": {"reader": "floodevent", "path": "projects/songliao/event"},
}


def resolve_data_path(
    dataset_id: str,
    *,
    ctx: Optional[ResolverContext] = None,
    **deprecated_kwargs,
) -> str:
    """Thin convenience wrapper — see hydrodataset.configs.data_resolver.

    Pre-injects HDS_DATASETS and HDS_READER_ALIASES into a ResolverContext,
    then delegates to hydrodataset's canonical resolver.
    """
    if ctx is None:
        ctx = ResolverContext(
            extra_registry_dicts=[HDS_DATASETS],
            extra_reader_aliases=HDS_READER_ALIASES,
        )
    return _hd_resolve_data_path(dataset_id, ctx=ctx, **deprecated_kwargs)
```

Key conventions:
- Build a `ResolverContext` with pre-injected data; pass it via `ctx=`
- Docstrings are minimal when wrapping — point to the canonical function
- No private helper functions that duplicate hydrodataset logic
- `Path(str)` normalization at entry points only (never `Path(Path(...))`)
- Do NOT pass `**kwargs` directly to hydrodataset — always use `ctx=`

---

## Testing Strategy

### Test levels

| Level | What | Where |
|-------|------|-------|
| Unit | `resolve_data_path` with mock storage | `hydrodataset/tests/test_configs_resolver.py` |
| Unit | Registry cascade (default → extra → YAML) | `hydrodataset/tests/test_configs_resolver.py` |
| Integration | End-to-end resolution with real paths | Existing per-dataset tests |
| Cross-package | hydrodatasource can resolve its datasets via hydrodataset | `hydrodatasource/tests/` |
| Cross-package | hydromodel's resolve_data_cfgs delegates correctly | `hydromodel/tests/` |

### TDD approach for each change:

1. Write test → RED
2. Minimal implementation → GREEN
3. Refactor → verify still GREEN
4. Coverage check → ≥80%

### Key test cases to add:

- `test_hydrodatasource_resolve_via_hydrodataset` — verify hydrodatasource's convenience calls hydrodataset
- `test_hydromodel_delegates_cloud_to_hydrodataset` — verify `_build_cloud_uri` is NOT reimplemented
- `test_layered_storage_in_hydrodataset` — verify project `.hydro_setting.yml` overrides user `~/hydro_setting.yml`

---

## Boundaries

### Always do:
- Run the full test suite before claiming a package is done
- Keep reader aliases and dataset entries as plain dicts (no classes, no factories)
- Delegate URI construction to hydrodataset — never reimplement `_build_cloud_uri`
- Use `extra_registry_dicts` + `extra_reader_aliases` for injection (proven pattern)

### Ask first:
- Adding new parameters to `resolve_data_path` (the signature is the contract)
- Changing the registry cascade order (default → extra → YAML)
- Moving logic FROM hydrodataset to another package

### Never do:
- Add `_build_cloud_uri` or equivalent S3 construction in any package except hydrodataset
- Add `_validate_relative_path` or equivalent validation in any package except hydrodataset
- Import resolution logic from hydromodel or torchhydro into hydrodataset (no circular deps)
- Hardcode storage paths in library code (always from settings or parameters)

---

## Success Criteria

- [ ] `_build_cloud_uri` exists exactly once: in `hydrodataset/configs/data_resolver.py`
- [ ] `_validate_relative_path` exists exactly once: in `hydrodataset/configs/data_resolver.py`
- [ ] `_load_yaml` exists exactly once: in `hydrodataset/configs/data_resolver.py`
- [ ] hydromodel's `data_resolver.py` is < 100 lines (down from 352)
- [ ] hydrodatasource's `data_resolver.py` is < 120 lines (down from 169)
- [ ] All 4 packages' test suites pass
- [ ] `grep -r "_build_cloud_uri\|_validate_relative_path\|_load_yaml"` returns hits ONLY in hydrodataset
- [ ] A researcher can `from hydrodataset import resolve_data_path; resolve_data_path("camels_us")` and get the right path

---

## Resolved Decisions

1. **hydrodatasource keeps its thin `resolve_data_path` re-export** — it pre-injects `HDS_DATASETS` + `_HDS_READER_ALIASES`. hydrodatasource has broader scope and may evolve into a dataset registry hub. The wrapper is ~5 lines of delegation, not logic duplication.
2. **Layered storage moves to hydrodataset** — `_load_storage` (merge `~/hydro_setting.yml` → `{project}/.hydro_setting.yml`) becomes the default behavior in hydrodataset's `resolve_data_path` for ALL callers. hydromodel's version is deleted.
3. **Immediate removal** of all duplicated logic. No deprecation warnings. `_build_cloud_uri`, `_load_yaml`, inline path validation — all deleted from hydromodel/hydrodatasource/torchhydro in one pass.
4. **Remove torchhydro's `resolve_dataset_uri`** — callers use hydrodataset (or hydrodatasource for HDS datasets) directly. `normalize_source_cfgs` stays as a config-format migration shim.
