import os
import xarray as xr
from hydrodataset import HydroDataset
from tqdm import tqdm
import numpy as np
from water_datasets import CAMELS_LUX


class CamelsLux(HydroDataset):
    """CAMELS_LUX dataset class extending RainfallRunoff.

    This class provides access to the CAMELS_LUX dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    def __init__(self, data_path, region=None, download=False, cache_path=None):
        """Initialize CAMELS_LUX dataset.

        Args:
            data_path: Path to the CAMELS_LUX data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
            cache_path: Path to the cache directory
        """
        super().__init__(data_path, cache_path=cache_path)
        self.region = region
        self.download = download
        self.aqua_fetch = CAMELS_LUX(data_path)

    @property
    def _attributes_cache_filename(self):
        return "camels_lux_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camels_lux_timeseries.nc"

    @property
    def default_t_range(self):
        return ["2004-01-01", "2021-12-31"]





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
            "mm",  # q_mm_obs
            "none",  # Qflag
            "mm",  # pcp_mm_radar
            "mm/5Min/1x1km",  # RR_min_rad
            "mm/5Min/1x1km",  # RR_max_rad
            "none",  # RR_flag_rad
            "mm",  # pcp_mm_station
            "mm",  # pcp_mm_era5
            "°C",  # airtemp_C_mean
            "mm",  # pet_mm_oudin
            "mm",  # pet_mm_pm
            "J/kg",  # cape
            "J/kg",  # cin
            "°C",  # kx
            "kg/kg",  # spechum_gkg
            "%",  # rh_%
            "kg/m^2",  # tcwv
            "m/s",  # windspeed_mps
            "m/s",  # lls
            "m/s",  # dls
            "m^3/m^3",  # sml1
            "m^3/m^3",  # sml2
            "m^3/m^3",  # sml3
            "m^3/m^3",  # sml4
        ]