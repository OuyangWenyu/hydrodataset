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
CAMELS_PE_PATH = _try_resolve("camels_pe")
CAMELS_LUX_PATH = _try_resolve("camels_lux")
CAMELS_NZ_PATH = _try_resolve("camels_nz")
CAMELS_DE_PATH = _try_resolve("camels_de")
CAMELS_FR_PATH = _try_resolve("camels_fr")
CAMELS_CH_PATH = _try_resolve("camels_ch")
CAMELS_US_PATH = _try_resolve("camels_us")
CAMELS_IND_PATH = _try_resolve("camels_ind")
CAMELS_BR_PATH = _try_resolve("camels_br")

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
needs_camels_pe = _needs("camels_pe", CAMELS_PE_PATH)
needs_camels_lux = _needs("camels_lux", CAMELS_LUX_PATH)
needs_camels_nz = _needs("camels_nz", CAMELS_NZ_PATH)
needs_camels_de = _needs("camels_de", CAMELS_DE_PATH)
needs_camels_fr = _needs("camels_fr", CAMELS_FR_PATH)
needs_camels_ch = _needs("camels_ch", CAMELS_CH_PATH)
needs_camels_us = _needs("camels_us", CAMELS_US_PATH)
needs_camels_ind = _needs("camels_ind", CAMELS_IND_PATH)
needs_camels_br = _needs("camels_br", CAMELS_BR_PATH)

# ── CARAVAN datasets ──────────────────────────────────────────────────

CARAVAN_PATH = _try_resolve("caravan")
CARAVAN_DK_PATH = _try_resolve("caravan_dk")
GRDC_CARAVAN_PATH = _try_resolve("grdc_caravan")

needs_caravan = _needs("caravan", CARAVAN_PATH)
needs_caravan_dk = _needs("caravan_dk", CARAVAN_DK_PATH)
needs_grdc_caravan = _needs("grdc_caravan", GRDC_CARAVAN_PATH)

# ── LamaH datasets ────────────────────────────────────────────────────

LAMAH_CE_PATH = _try_resolve("lamah_ce")
LAMAH_ICE_PATH = _try_resolve("lamah_ice")

needs_lamah_ce = _needs("lamah_ce", LAMAH_CE_PATH)
needs_lamah_ice = _needs("lamah_ice", LAMAH_ICE_PATH)

# ── Other public datasets ──────────────────────────────────────────────

BULL_PATH = _try_resolve("bull")
HYSETS_PATH = _try_resolve("hysets")
ESTREAMS_PATH = _try_resolve("estreams")
SIMBI_PATH = _try_resolve("simbi")

needs_bull = _needs("bull", BULL_PATH)
needs_hysets = _needs("hysets", HYSETS_PATH)
needs_estreams = _needs("estreams", ESTREAMS_PATH)
needs_simbi = _needs("simbi", SIMBI_PATH)
