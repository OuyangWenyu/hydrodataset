"""Shared dataset path resolution for all test modules.

Each dataset gets a single resolver call (cached at module level) and a
corresponding ``needs_*`` skip marker. Test files import the names they need
instead of repeating the try/except boilerplate.

Usage::

    from tests._paths import CAMELSH_PATH, needs_camelsh, DATA_ROOT

    @needs_camelsh
    @skip_if_ci
    def test_something():
        ds = Camelsh(CAMELSH_PATH)
        csv_path = os.path.join(str(DATA_ROOT), "CAMELSH", ...)
"""

from hydrodataset import resolve_data_path, DatasetResolutionError, get_local_root

# Root data directory from user settings (absolute path)
DATA_ROOT = get_local_root()

# ── Private helpers ────────────────────────────────────────────────────


def _try_resolve(dataset_id: str):
    """Resolve *dataset_id* to an absolute path, or return ``None``."""
    try:
        return resolve_data_path(dataset_id)
    except DatasetResolutionError:
        return None


def _needs(dataset_id: str, resolved_path):
    """Build a ``pytest.mark.skipif`` for *dataset_id*."""
    import pytest

    return pytest.mark.skipif(
        resolved_path is None,
        reason=(
            f"{dataset_id} data not available -- "
            f"configure storage.local.root in ~/hydro_setting.yml"
        ),
    )


# ── CAMELS / CAMELSH datasets (all aqua_fetch-based, path: . in datasets.yml) ──

CAMELSH_PATH = _try_resolve("camelsh")
CAMELS_AUS_PATH = _try_resolve("camels_aus")
CAMELS_CL_PATH = _try_resolve("camels_cl")
CAMELS_DK_PATH = _try_resolve("camels_dk")
CAMELS_COL_PATH = _try_resolve("camels_col")
CAMELS_SE_PATH = _try_resolve("camels_se")
CAMELSH_KR_PATH = _try_resolve("camelsh_kr")
CAMELS_GB_PATH = _try_resolve("camels_gb")
CAMELS_FI_PATH = _try_resolve("camels_fi")
CAMELS_LUX_PATH = _try_resolve("camels_lux")
CAMELS_NZ_PATH = _try_resolve("camels_nz")
CAMELS_DE_PATH = _try_resolve("camels_de")
CAMELS_FR_PATH = _try_resolve("camels_fr")
CAMELS_CH_PATH = _try_resolve("camels_ch")
CAMELS_US_PATH = _try_resolve("camels_us")

# ── Skip markers (one per dataset) ─────────────────────────────────────

needs_camelsh = _needs("camelsh", CAMELSH_PATH)
needs_camels_aus = _needs("camels_aus", CAMELS_AUS_PATH)
needs_camels_cl = _needs("camels_cl", CAMELS_CL_PATH)
needs_camels_dk = _needs("camels_dk", CAMELS_DK_PATH)
needs_camels_col = _needs("camels_col", CAMELS_COL_PATH)
needs_camels_se = _needs("camels_se", CAMELS_SE_PATH)
needs_camelsh_kr = _needs("camelsh_kr", CAMELSH_KR_PATH)
needs_camels_gb = _needs("camels_gb", CAMELS_GB_PATH)
needs_camels_fi = _needs("camels_fi", CAMELS_FI_PATH)
needs_camels_lux = _needs("camels_lux", CAMELS_LUX_PATH)
needs_camels_nz = _needs("camels_nz", CAMELS_NZ_PATH)
needs_camels_de = _needs("camels_de", CAMELS_DE_PATH)
needs_camels_fr = _needs("camels_fr", CAMELS_FR_PATH)
needs_camels_ch = _needs("camels_ch", CAMELS_CH_PATH)
needs_camels_us = _needs("camels_us", CAMELS_US_PATH)
