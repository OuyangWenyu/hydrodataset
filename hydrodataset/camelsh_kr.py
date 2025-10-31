import os
import xarray as xr
from hydrodataset import HydroDataset, StandardVariable
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

    # not find information of features
    _subclass_static_definitions = {
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
    }

    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "obs",
            "sources": {
                "obs": {"specific_name": "q_cms_obs", "unit": "m^3/s"},
            },
        },
        StandardVariable.WATER_LEVEL: {
            "default_source": "obs",
            "sources": {
                "obs": {"specific_name": "water_level", "unit": "m"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "total_precipitation", "unit": "mm/day"},
                "obs": {"specific_name": "precip_obs", "unit": "mm/day"},
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "temperature_2m", "unit": "°C"},
                "obs": {"specific_name": "air_temp_obs", "unit": "°C"},
                "dewpoint": {"specific_name": "dewpoint_temperature_2m", "unit": "°C"},
            },
        },
        StandardVariable.VAPOR_PRESSURE: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "dewpoint_temperature_2m", "unit": "°C"},
            },
        },
        StandardVariable.SNOW_DEPTH: {
            "default_source": "era5_depth",
            "sources": {
                "era5_depth": {"specific_name": "snow_depth", "unit": "m"},
            },
        },
        StandardVariable.SNOW_COVER: {
            "default_source": "era5_cover",
            "sources": {
                "era5_cover": {"specific_name": "snow_cover", "unit": "fraction"},
            },
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "potential_evaporation", "unit": "mm/day"},
            },
        },
        StandardVariable.U_WIND_SPEED: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "u_component_of_wind_10m", "unit": "m/s"},
            },
        },
        StandardVariable.V_WIND_SPEED: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "v_component_of_wind_10m", "unit": "m/s"},
            },
        },
        StandardVariable.WIND_SPEED: {
            "default_source": "obs_speed",
            "sources": {
                "obs_speed": {"specific_name": "wind_sp_obs", "unit": "m/s"},
            },
        },
        StandardVariable.WIND_DIR: {
            "default_source": "obs_dir",
            "sources": {
                "obs_dir": {"specific_name": "wind_dir_obs", "unit": "degree"},
            },
        },
        StandardVariable.SURFACE_PRESSURE: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "surface_pressure", "unit": "Pa"},
            },
        },
        StandardVariable.THERMAL_RADIATION: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "surface_net_thermal_radiation",
                    "unit": "W/m^2",
                },
            },
        },
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "surface_net_solar_radiation",
                    "unit": "W/m^2",
                },
            },
        },
    }
