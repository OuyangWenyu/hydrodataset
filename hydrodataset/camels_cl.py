import os
import xarray as xr
import numpy as np
from aqua_fetch import CAMELS_CL
from hydrodataset import HydroDataset
from tqdm import tqdm


class CamelsCl(HydroDataset):
    """CAMELS_CL dataset class extending RainfallRunoff.

    This class provides access to the CAMELS_CL dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    def __init__(self, data_path, region=None, download=False, cache_path=None):
        """Initialize CAMELS_CL dataset.

        Args:
            data_path: Path to the CAMELS_CL data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
            cache_path: Path to the cache directory
        """
        super().__init__(data_path, cache_path=cache_path)
        self.region = region
        self.download = download
        self.aqua_fetch = CAMELS_CL(data_path)

    @property
    def _attributes_cache_filename(self):
        return "camels_cl_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camels_cl_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1913-02-15", "2018-03-09"]





    def _get_attribute_units(self):
        return {
            # 地形特征
            "dis_m3_": "m^3/s",
            "run_mm_": "millimeter",
            "inu_pc_": "percent",
            "lka_pc_": "1e-1 * percent",
            "lkv_mc_": "1e6 * m^3",
            "rev_mc_": "1e6 * m^3",
            "dor_pc_": "percent (x10)",
            "ria_ha_": "hectares",
            "riv_tc_": "1e3 * m^3",
            "gwt_cm_": "centimeter",
            "ele_mt_": "meter",
            "slp_dg_": "1e-1 * degree",
            "sgr_dk_": "decimeter/km",
            "clz_cl_": "dimensionless",
            "cls_cl_": "dimensionless",
            "tmp_dc_": "degree_Celsius",
            "pre_mm_": "millimeters",
            "pet_mm_": "millimeters",
            "aet_mm_": "millimeters",
            "ari_ix_": "1e-2",
            "cmi_ix_": "1e-2",
            "snw_pc_": "percent",
            "glc_cl_": "dimensionless",
            "glc_pc_": "percent",
            "pnv_cl_": "dimensionless",
            "pnv_pc_": "percent",
            "wet_cl_": "dimensionless",
            "wet_pc_": "percent",
            "for_pc_": "percent",
            "crp_pc_": "percent",
            "pst_pc_": "percent",
            "ire_pc_": "percent",
            "gla_pc_": "percent",
            "prm_pc_": "percent",
            "pac_pc_": "percent",
            "tbi_cl_": "dimensionless",
            "tec_cl_": "dimensionless",
            "fmh_cl_": "dimensionless",
            "fec_cl_": "dimensionless",
            "cly_pc_": "percent",
            "slt_pc_": "percent",
            "snd_pc_": "percent",
            "soc_th_": "tonne/hectare",
            "swc_pc_": "percent",
            "lit_cl_": "dimensionless",
            "kar_pc_": "percent",
            "ero_kh_": "kg/hectare/year",
            "pop_ct_": "1e3",
            "ppd_pk_": "1/km^2",
            "urb_pc_": "percent",
            "nli_ix_": "1e-2",
            "rdd_mk_": "meter/km^2",
            "hft_ix_": "1e-1",
            "gad_id_": "dimensionless",
            "gdp_ud_": "dimensionless",
            "hdi_ix_": "1e-3",
        }

    def _get_timeseries_units(self):
        return [
            "m^3/s",  # q_cms_obs
            "mm/day",  # q_mm_obs
            "mm/day",  # pcp_mm_cr2met
            "mm/day",  # pcp_mm_chirps
            "mm/day",  # pcp_mm_mswep
            "mm/day",  # pcp_mm_tmpa
            "°C",  # airtemp_C_min
            "°C",  # airtemp_C_max
            "°C",  # airtemp_C_mean
            "mm/day",  # pet_mm_modis
            "mm/day",  # pet_mm_hargreaves
            "mm",  # swe
        ]