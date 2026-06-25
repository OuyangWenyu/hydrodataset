# Loop Plan: Unified Data Resolution — hydrodataset layered storage

**Pattern**: sequential
**Mode**: safe (TDD with quality gates)
**Branch**: feat/unified-data-path-resolution
**Created**: 2026-06-25

## Stop Condition

Loop stops after Step 3 completes AND all tests pass. Or on any step failure (RED that can't be fixed within the step).

## Quality Gates (per step)

- [ ] Tests pass (GREEN)
- [ ] No regressions in existing tests
- [ ] Code follows surrounding conventions

## Steps

| # | Step | Chain | Gate |
|---|------|-------|------|
| 1 | Add `_load_storage()` | tdd-guide, python-reviewer | New tests pass + existing tests pass |
| 2 | Add `storage` param + update branches | tdd-guide, python-reviewer | New tests pass + all 14 resolver tests pass |
| 3 | Full regression + refactor | tdd-guide, refactor-cleaner, python-reviewer | All 126+ tests pass, no duplicate logic |

## Orchestrate Commands

### Step 1

/ecc:orchestrate custom "ecc:tdd-guide,ecc:python-reviewer" "[Plan: docs/specs/unified-data-resolution-plan.md#step-1] Add _load_storage(project_root: Path) -> Dict function to hydrodataset/configs/data_resolver.py that merges storage config from ~/hydro_setting.yml (user-level) and {project_root}/.hydro_setting.yml (project-level, overrides user); returns merged dict with local.root and s3 sections; uses existing _load_yaml helper; Acceptance: test_layered_storage_project_overrides_user passes; test_storage_empty_when_no_files passes; existing tests unchanged"

### Step 2

/ecc:orchestrate custom "ecc:tdd-guide,ecc:python-reviewer" "[Plan: docs/specs/unified-data-resolution-plan.md#step-2] Add optional storage: Optional[Dict[str, Any]] = None keyword-only parameter to resolve_data_path() in hydrodataset/configs/data_resolver.py; update local branch to use storage[\"local\"][\"root\"] when local_root is None and storage is provided; update cloud branch to use storage[\"s3\"] instead of get_storage_config() when storage is provided; when storage is None fall back to get_local_root()/get_storage_config(); Acceptance: test_storage_param_overrides_settings passes; test_storage_none_preserves_behavior passes; all 14 test_configs_resolver tests pass"

### Step 3

/ecc:orchestrate custom "ecc:tdd-guide,ecc:refactor-cleaner,ecc:python-reviewer" "[Plan: docs/specs/unified-data-resolution-plan.md#step-3] Run full test suite (pytest tests/ -v --tb=short); verify all 126+ tests pass; refactor resolve_data_path to ensure get_local_root()/get_storage_config() only called through storage=None fallback; verify _build_cloud_uri exists only in hydrodataset/configs/data_resolver.py; Acceptance: full test suite passes; grep confirms no duplicate path-building logic; _load_storage is single entry point for storage config"
