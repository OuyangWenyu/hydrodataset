from hydrodataset import HydroDataset, StandardVariable
from aqua_fetch import CAMELS_GB


class CamelsGb(HydroDataset):
    """CAMELS_GB dataset class extending RainfallRunoff.

    This class provides access to the CAMELS_GB dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    def __init__(self, data_path, region=None, download=False):
        """Initialize CAMELS_GB dataset.

        Args:
            data_path: Path to the CAMELS_GB data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
        """
        super().__init__(data_path)
        self.region = region
        self.download = download
        self.aqua_fetch = CAMELS_GB(data_path)

    @property
    def _attributes_cache_filename(self):
        return "camels_gb_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camels_gb_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1970-10-01", "2015-09-30"]

    # get the information of features from dataset file"CAMELSGB_EIDC_SupportingDocumentation"
    _subclass_static_definitions = {
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
        "gauge_lat": {"specific_name": "lat", "unit": "degree"},
        "gauge_lon": {"specific_name": "long", "unit": "degree"},
        "elev_mean": {"specific_name": "elev_mean", "unit": "m"},
        "pet_mean": {"specific_name": "pet_mean", "unit": "mm/day"},
    }

    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "hydrological",
            "sources": {
                "hydrological": {"specific_name": "q_cms_obs", "unit": "m^3/s"},
                "depth_based": {"specific_name": "q_mm_obs", "unit": "mm/day"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "meteorological",
            "sources": {
                "meteorological": {"specific_name": "pcp_mm", "unit": "mm/day"},
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "meteorological",
            "sources": {
                "meteorological": {"specific_name": "airtemp_C_mean", "unit": "°C"},
            },
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "meteorological",
            "sources": {
                "meteorological": {"specific_name": "pet_mm", "unit": "mm/day"},
                "with_interception": {
                    "specific_name": "pet_mm_intercep",
                    "unit": "mm/day",
                },
            },
        },
        StandardVariable.RELATIVE_HUMIDITY: {
            "default_source": "meteorological",
            "sources": {
                "meteorological": {"specific_name": "rh_%", "unit": "g/kg"},
            },
        },
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "meteorological",
            "sources": {
                "meteorological": {"specific_name": "solrad_wm2", "unit": "W/m^2"},
            },
        },
        StandardVariable.LONGWAVE_SOLAR_RADIATION: {
            "default_source": "meteorological",
            "sources": {
                "meteorological": {"specific_name": "lwsolrad_wm2", "unit": "W/m^2"},
            },
        },
        StandardVariable.WIND_SPEED: {
            "default_source": "meteorological",
            "sources": {
                "meteorological": {"specific_name": "windspeed_mps", "unit": "m/s"},
            },
        },
    }

   