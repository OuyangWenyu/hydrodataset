from aqua_fetch import Bull
from hydrodataset import HydroDataset, StandardVariable


class BULL(HydroDataset):
    """Bull dataset class extending RainfallRunoff.

    This class provides access to the Bull dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    def __init__(self, data_path, region=None, download=False):
        """Initialize Bull dataset.

        Args:
            data_path: Path to the Bull data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
            cache_path: Path to the cache directory
        """
        super().__init__(data_path)
        self.region = region
        self.download = download
        self.aqua_fetch = Bull(data_path)

    @property
    def _attributes_cache_filename(self):
        return "bull_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "bull_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1951-01-02", "2021-12-31"]

    _subclass_static_definitions = {
        "area": {"specific_name": "area_km2", "unit": "km^2"},
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
    }

    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "streamflow", "unit": "m^3/s"},
                "q_cms": {"specific_name": "q_cms_obs", "unit": "m^3/s"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "aemet",
            "sources": {
                "aemet": {"specific_name": "pcp_mm_aemet", "unit": "mm/day"},
                "bull": {"specific_name": "pcp_mm_bull", "unit": "mm/day"},
                "era5land": {"specific_name": "pcp_mm_era5land", "unit": "mm/day"},
                "emo1arc": {"specific_name": "pcp_mm_emo1arc", "unit": "mm/day"},
            },
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "aemet",
            "sources": {
                "aemet": {"specific_name": "airtemp_c_aemet_max", "unit": "°C"},
                "era5land": {"specific_name": "airtemp_c_era5land_max", "unit": "°C"},
                "emo1arc": {"specific_name": "airtemp_c_emo1arc_max", "unit": "°C"},
                "2m": {"specific_name": "airtemp_c_2m_max", "unit": "°C"},
                "dewpoint": {"specific_name": "dptemp_c_max", "unit": "°C"},
            },
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "aemet",
            "sources": {
                "aemet": {"specific_name": "airtemp_c_aemet_min", "unit": "°C"},
                "era5land": {"specific_name": "airtemp_c_era5land_min", "unit": "°C"},
                "emo1arc": {"specific_name": "airtemp_c_emo1arc_min", "unit": "°C"},
                "2m": {"specific_name": "airtemp_c_2m_min", "unit": "°C"},
                "dewpoint": {"specific_name": "dptemp_c_min", "unit": "°C"},
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "aemet",
            "sources": {
                "aemet": {"specific_name": "airtemp_c_mean_aemet", "unit": "°C"},
                "era5land": {"specific_name": "airtemp_c_mean_era5land", "unit": "°C"},
                "emo1arc": {"specific_name": "airtemp_c_mean_emo1arc", "unit": "°C"},
                "2m": {"specific_name": "airtemp_c_mean_2m", "unit": "°C"},
                "dewpoint": {"specific_name": "dptemp_c_mean", "unit": "°C"},
            },
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "aemet",
            "sources": {
                "aemet": {"specific_name": "pet_mm_aemet", "unit": "mm/day"},
                "era5land": {"specific_name": "pet_mm_era5land", "unit": "mm/day"},
                "emo1arc": {"specific_name": "pet_mm_emo1arc", "unit": "mm/day"},
            },
        },
        StandardVariable.EVAPORATION: {
            "default_source": "bull",
            "sources": {"bull": {"specific_name": "pevap_mm", "unit": "mm/day"}},
        },
        # Snow water equivalent - separate MIN and MAX as independent variables
        StandardVariable.SNOW_WATER_EQUIVALENT: {
            "default_source": "observations",
            "sources": {"observations": {"specific_name": "swe_mm", "unit": "mm"}},
        },
        StandardVariable.SNOW_WATER_EQUIVALENT_MIN: {
            "default_source": "observations",
            "sources": {"observations": {"specific_name": "swe_mm_min", "unit": "mm"}},
        },
        StandardVariable.SNOW_WATER_EQUIVALENT_MAX: {
            "default_source": "observations",
            "sources": {"observations": {"specific_name": "swe_mm_max", "unit": "mm"}},
        },
        # Solar radiation - separate MIN and MAX as independent variables
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "solrad_wm2", "unit": "W/m^2"}
            },
        },
        StandardVariable.SOLAR_RADIATION_MIN: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "solrad_wm2_min", "unit": "W/m^2"}
            },
        },
        StandardVariable.SOLAR_RADIATION_MAX: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "solrad_wm2_max", "unit": "W/m^2"}
            },
        },
        # Thermal radiation - separate MIN and MAX as independent variables
        StandardVariable.THERMAL_RADIATION: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "thermrad_wm2", "unit": "W/m^2"}
            },
        },
        StandardVariable.THERMAL_RADIATION_MIN: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "thermrad_wm2_min", "unit": "W/m^2"}
            },
        },
        StandardVariable.THERMAL_RADIATION_MAX: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "thermrad_wm2_max", "unit": "W/m^2"}
            },
        },
        # Surface pressure - separate MIN and MAX as independent variables
        StandardVariable.SURFACE_PRESSURE: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "surface_pressure_mean_bull",
                    "unit": "Pa",
                }
            },
        },
        StandardVariable.SURFACE_PRESSURE_MIN: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "surface_pressure_min_bull",
                    "unit": "Pa",
                }
            },
        },
        StandardVariable.SURFACE_PRESSURE_MAX: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "surface_pressure_max_bull",
                    "unit": "Pa",
                }
            },
        },
        # U wind speed - separate MIN and MAX as independent variables
        StandardVariable.U_WIND_SPEED: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "windspeedu_mps_mean_10m",
                    "unit": "m/s",
                }
            },
        },
        StandardVariable.U_WIND_SPEED_MIN: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "windspeedu_mps_min_10m",
                    "unit": "m/s",
                }
            },
        },
        StandardVariable.U_WIND_SPEED_MAX: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "windspeedu_mps_max_10m",
                    "unit": "m/s",
                }
            },
        },
        # V wind speed - separate MIN and MAX as independent variables
        StandardVariable.V_WIND_SPEED: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "windspeedv_mps_mean_10m",
                    "unit": "m/s",
                }
            },
        },
        StandardVariable.V_WIND_SPEED_MIN: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "windspeedv_mps_min_10m",
                    "unit": "m/s",
                }
            },
        },
        StandardVariable.V_WIND_SPEED_MAX: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "windspeedv_mps_max_10m",
                    "unit": "m/s",
                }
            },
        },
        # Volumetric soil water layer 1 - separate MIN and MAX as independent variables
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER1: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "volumetric_soil_water_layer_1_mean_bull",
                    "unit": "m^3/m^3",
                }
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER1_MIN: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "volumetric_soil_water_layer_1_min_bull",
                    "unit": "m^3/m^3",
                }
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER1_MAX: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "volumetric_soil_water_layer_1_max_bull",
                    "unit": "m^3/m^3",
                }
            },
        },
        # Volumetric soil water layer 2 - separate MIN and MAX as independent variables
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER2: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "volumetric_soil_water_layer_2_mean_bull",
                    "unit": "m^3/m^3",
                }
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER2_MIN: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "volumetric_soil_water_layer_2_min_bull",
                    "unit": "m^3/m^3",
                }
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER2_MAX: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "volumetric_soil_water_layer_2_max_bull",
                    "unit": "m^3/m^3",
                }
            },
        },
        # Volumetric soil water layer 3 - separate MIN and MAX as independent variables
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER3: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "volumetric_soil_water_layer_3_mean_bull",
                    "unit": "m^3/m^3",
                }
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER3_MIN: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "volumetric_soil_water_layer_3_min_bull",
                    "unit": "m^3/m^3",
                }
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER3_MAX: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "volumetric_soil_water_layer_3_max_bull",
                    "unit": "m^3/m^3",
                }
            },
        },
        # Volumetric soil water layer 4 - separate MIN and MAX as independent variables
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER4: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "volumetric_soil_water_layer_4_mean_bull",
                    "unit": "m^3/m^3",
                }
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER4_MIN: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "volumetric_soil_water_layer_4_min_bull",
                    "unit": "m^3/m^3",
                }
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER4_MAX: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "volumetric_soil_water_layer_4_max_bull",
                    "unit": "m^3/m^3",
                }
            },
        },
    }
