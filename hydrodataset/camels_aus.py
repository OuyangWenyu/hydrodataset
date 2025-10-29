import os
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from aqua_fetch import CAMELS_AUS
from hydrodataset import HydroDataset,StandardVariable


class CamelsAus(HydroDataset):
    """CAMELS_AUS dataset class extending RainfallRunoff.

    This class provides access to the CAMELS_AUS dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    def __init__(self, data_path, region=None, download=False):
        """Initialize CAMELS_AUS dataset.

        Args:
            data_path: Path to the CAMELS_AUS data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
            cache_path: Path to the cache directory
        """
        super().__init__(data_path)
        self.region = region
        self.download = download
        self.aqua_fetch = CAMELS_AUS(data_path)

    @property
    def _attributes_cache_filename(self):
        return "camels_aus_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camels_aus_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1950-01-01", "2022-03-31"]


    _subclass_static_definitions = {
        "p_mean": {"specific_name": "p_mean", "unit": "mm"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
    }
    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "bom",
            "sources": {
                "bom": {"specific_name": "q_cms_obs", "unit": "m^3/s"},
                "gr4j": {"specific_name": "streamflow_mld_inclinfilled", "unit": "m^3/s"},
                "depth_based": {"specific_name": "q_mm_obs", "unit": "m^3/s"}
            },
        },
        
        StandardVariable.EVAPOTRANSPIRATION: {
            "default_source": "silo_morton",
            "sources": {
                "silo_morton": {"specific_name": "aet_mm_silo_morton", "unit": "mm/day"},
            },
        },
        
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "silo_morton",
            "sources": {
                "silo_morton": {"specific_name": "et_morton_wet_silo", "unit": "mm/day"},
                "silo_morton_point": {"specific_name": "aet_mm_silo_morton_point", "unit": "mm/day"},
                "silo_short_crop": {"specific_name": "aet_mm_silo_short_crop", "unit": "mm/day"},
                "silo_tall_crop": {"specific_name": "aet_mm_silo_tall_crop", "unit": "mm/day"},
            },
        },
        
        StandardVariable.EVAPORATION: {
            "default_source": "silo_morton_lake",
            "sources": {
                "silo_morton_lake": {"specific_name": "evap_morton_lake_silo", "unit": "mm/day"},
                "silo_pan": {"specific_name": "evap_pan_silo", "unit": "mm/day"},
                "silo_syn": {"specific_name": "evap_syn_silo", "unit": "mm/day"}
            },
        },
        
        StandardVariable.PRECIPITATION: {
            "default_source": "agcd",
            "sources": {
                "agcd": {"specific_name": "pcp_mm_agcd", "unit": "mm/day"},
                "silo": {"specific_name": "pcp_mm_silo", "unit": "mm/day"},
                "agcd_var": {"specific_name": "precipitation_var_agcd", "unit": "mm^2/day^2"}
            },
        },
        
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "agcd",
            "sources": {
                "agcd": {"specific_name": "airtemp_c_agcd_max", "unit": "°C"},
                "silo": {"specific_name": "airtemp_c_silo_max", "unit": "°C"}
            },
        },
        
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "agcd",
            "sources": {
                "agcd": {"specific_name": "airtemp_c_agcd_min", "unit": "°C"},
                "silo": {"specific_name": "airtemp_c_silo_min", "unit": "°C"}
            },
        },
        
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "silo",
            "sources": {
                "silo": {"specific_name": "airtemp_c_mean_silo", "unit": "°C"},
                "agcd": {"specific_name": "airtemp_c_mean_agcd", "unit": "°C"}
            },
        },
        
        StandardVariable.VAPOR_PRESSURE: {
            "default_source": "agcd_h09",
            "sources": {
                "agcd_h09": {"specific_name": "vp_hpa_agcd_h09", "unit": "hPa"},
                "agcd_h15": {"specific_name": "vp_hpa_agcd_h15", "unit": "hPa"},
                "silo": {"specific_name": "vp_hpa_silo", "unit": "hPa"},
                "silo_deficit": {"specific_name": "vp_deficit_silo", "unit": "hPa"}
            },
        },
        
        StandardVariable.RELATIVE_HUMIDITY: {
            "default_source": "silo_tmax",
            "sources": {
                "silo_tmax": {"specific_name": "rh__silo_tmax", "unit": "%"},
                "silo_tmin": {"specific_name": "rh__silo_tmin", "unit": "%"}
            },
        },
        
        StandardVariable.SEA_LEVEL_PRESSURE: {
            "default_source": "silo",
            "sources": {
                "silo": {"specific_name": "mslp_silo", "unit": "hPa"}
            },
        },
        
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "silo",
            "sources": {
                "silo": {"specific_name": "solrad_wm2_silo", "unit": "W/m^2"}
            },
        }
    }

'''
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
            "mm^3/s",  # q_cms_obs ML/day->mm^3/s,×1.1574 × 10⁻2
            "ML/day",  # streamflow_MLd
            "mm/day",  # q_mm_obs
            "mm/day",  # aet_mm_silo_morton
            "mm/day",  # aet_mm_silo_morton_point
            "mm/day",  # et_morton_wet_SILO
            "mm/day",  # aet_mm_silo_short_crop
            "mm/day",  # aet_mm_silo_tall_crop
            "mm/day",  # evap_morton_lake_SILO
            "mm/day",  # evap_pan_SILO
            "mm/day",  # evap_syn_SILO
            "mm/day",  # pcp_mm_agcd
            "mm/day",  # pcp_mm_silo
            "mm^2/d^2 ",  # precipitation_var_AGCD
            "°C",  # airtemp_C_agcd_max
            "°C",  # airtemp_C_agcd_min
            "hPa",  # vp_hpa_agcd_h09
            "hPa",  # vp_hpa_agcd_h15
            "hPa",  # mslp_SILO
            "MJ/m²",  # solrad_wm2_silo
            "%",  # rh_%_silo_tmax
            "%",  # rh_%_silo_tmin
            "°C",  # airtemp_C_silo_max
            "°C",  # airtemp_C_silo_min
            "hPa",  # vp_deficit_SILO
            "hPa",  # vp_hpa_silo
            "°C",  # airtemp_C_mean_silo
            "°C",  # airtemp_C_mean_agcd
        ]

    subclass_variable_name_map = {}
'''