import os
import xarray as xr
from hydrodataset import HydroDataset
from tqdm import tqdm
import numpy as np
from aqua_fetch import CAMELS_SK


class CamelshKr(HydroDataset):
    """CAMELSH_KR dataset class extending RainfallRunoff.

    This class provides access to the CAMELSH_KR dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    def __init__(self, data_path, region=None, download=False, cache_path=None):
        """Initialize CAMELSH_KR dataset.

        Args:
            data_path: Path to the CAMELSH_KR data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
            cache_path: Path to the cache directory
        """
        super().__init__(data_path, cache_path=cache_path)
        self.region = region
        self.download = download
        # In aqua_fetch, CAMELS_SK is the alias of CAMELSH_KR
        self.aqua_fetch = CAMELS_SK(data_path)

    @property
    def _attributes_cache_filename(self):
        return "camels_sk_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camels_sk_timeseries.nc"

    @property
    def default_t_range(self):
        return ["2000-01-01", "2019-12-31"]





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
            "unknown",  # total_precipitation
            "unknown",  # temperature_2m
            "unknown",  # dewpoint_temperature_2m
            "unknown",  # snow_cover
            "unknown",  # snow_depth
            "unknown",  # potential_evaporation
            "unknown",  # u_component_of_wind_10m
            "unknown",  # v_component_of_wind_10m
            "unknown",  # surface_pressure
            "unknown",  # surface_net_thermal_radiation
            "unknown",  # surface_net_solar_radiation
            "unknown",  # precip_obs
            "unknown",  # air_temp_obs
            "unknown",  # wind_dir_obs
            "unknown",  # wind_sp_obs
            "unknown",  # q_cms_obs
            "unknown",  # water_level
        ]


    def read_mean_prcp(self, gage_id_lst, unit="mm/d") -> xr.Dataset:
        data_output_ds_ = self.read_attr_xrdataset(
            gage_id_lst,
            ['p_mean'],
        )
        data_output_ds_ = (
            data_output_ds_
            .rename({'STAID': 'basin'})  # 重命名 STAID 为 basin
        )
        return data_output_ds_