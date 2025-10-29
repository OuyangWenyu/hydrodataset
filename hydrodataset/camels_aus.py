import os
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from aqua_fetch import CAMELS_AUS
from hydrodataset import HydroDataset, StandardVariable


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
                "gr4j": {
                    "specific_name": "streamflow_mld_inclinfilled",
                    "unit": "m^3/s",
                },
                "depth_based": {"specific_name": "q_mm_obs", "unit": "m^3/s"},
            },
        },
        StandardVariable.EVAPOTRANSPIRATION: {
            "default_source": "silo_morton",
            "sources": {
                "silo_morton": {
                    "specific_name": "aet_mm_silo_morton",
                    "unit": "mm/day",
                },
            },
        },
        # For PET, AET and ET, the explanation is in the CAMELS_AUS paper, table 2.
        # table 2 in https://essd.copernicus.org/articles/13/3847/2021/#&gid=1&pid=1
        # But the specific names are not the same as the ones in the paper but same as the ones renamed by aqua_fetch.
        # https://github.com/hyex-research/AquaFetch/blob/143c1578fcf18dd6f3a47ba1f2214b089e6e47a9/aqua_fetch/rr/_camels.py#L905C1-L908C93
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "silo_morton",
            "sources": {
                "silo_morton": {
                    "specific_name": "et_morton_wet_silo",
                    "unit": "mm/day",
                },
                "silo_morton_point": {
                    "specific_name": "aet_mm_silo_morton_point",
                    "unit": "mm/day",
                },
                "silo_short_crop": {
                    "specific_name": "aet_mm_silo_short_crop",
                    "unit": "mm/day",
                },
                "silo_tall_crop": {
                    "specific_name": "aet_mm_silo_tall_crop",
                    "unit": "mm/day",
                },
            },
        },
        StandardVariable.EVAPORATION: {
            "default_source": "silo_morton_lake",
            "sources": {
                "silo_morton_lake": {
                    "specific_name": "evap_morton_lake_silo",
                    "unit": "mm/day",
                },
                "silo_pan": {"specific_name": "evap_pan_silo", "unit": "mm/day"},
                "silo_syn": {"specific_name": "evap_syn_silo", "unit": "mm/day"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "agcd",
            "sources": {
                "agcd": {"specific_name": "pcp_mm_agcd", "unit": "mm/day"},
                "silo": {"specific_name": "pcp_mm_silo", "unit": "mm/day"},
                "agcd_var": {
                    "specific_name": "precipitation_var_agcd",
                    "unit": "mm^2/day^2",
                },
            },
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "agcd",
            "sources": {
                "agcd": {"specific_name": "airtemp_c_agcd_max", "unit": "°C"},
                "silo": {"specific_name": "airtemp_c_silo_max", "unit": "°C"},
            },
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "agcd",
            "sources": {
                "agcd": {"specific_name": "airtemp_c_agcd_min", "unit": "°C"},
                "silo": {"specific_name": "airtemp_c_silo_min", "unit": "°C"},
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "silo",
            "sources": {
                "silo": {"specific_name": "airtemp_c_mean_silo", "unit": "°C"},
                "agcd": {"specific_name": "airtemp_c_mean_agcd", "unit": "°C"},
            },
        },
        StandardVariable.VAPOR_PRESSURE: {
            "default_source": "agcd_h09",
            "sources": {
                "agcd_h09": {"specific_name": "vp_hpa_agcd_h09", "unit": "hPa"},
                "agcd_h15": {"specific_name": "vp_hpa_agcd_h15", "unit": "hPa"},
                "silo": {"specific_name": "vp_hpa_silo", "unit": "hPa"},
                "silo_deficit": {"specific_name": "vp_deficit_silo", "unit": "hPa"},
            },
        },
        StandardVariable.RELATIVE_HUMIDITY: {
            "default_source": "silo_tmax",
            "sources": {
                "silo_tmax": {"specific_name": "rh__silo_tmax", "unit": "%"},
                "silo_tmin": {"specific_name": "rh__silo_tmin", "unit": "%"},
            },
        },
        StandardVariable.SEA_LEVEL_PRESSURE: {
            "default_source": "silo",
            "sources": {"silo": {"specific_name": "mslp_silo", "unit": "hPa"}},
        },
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "silo",
            "sources": {"silo": {"specific_name": "solrad_wm2_silo", "unit": "W/m^2"}},
        },
    }
