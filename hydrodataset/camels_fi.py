from typing import Optional

from hydrodataset import HydroDataset, StandardVariable
from aqua_fetch import CAMELS_FI


class CamelsFi(HydroDataset):
    """CAMELS_FI dataset class extending RainfallRunoff.

    This class provides access to the CAMELS_FI dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    def __init__(
        self, data_path: str, region: Optional[str] = None, download: bool = False
    ) -> None:
        """Initialize CAMELS_FI dataset.

        Args:
            data_path: Path to the CAMELS_FI data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
        """
        super().__init__(data_path)
        self.region = region
        self.download = download
        self.aqua_fetch = CAMELS_FI(data_path)

    @property
    def _attributes_cache_filename(self):
        return "camels_fi_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camels_fi_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1961-01-01", "2023-12-31"]

    # get the information of features from dataset file"support_document.pdf"
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
            "default_source": "SYKE",
            "sources": {
                "SYKE": {"specific_name": "q_cms_obs", "unit": "m^3/s"},
                "depth_based": {"specific_name": "q_mm_obs", "unit": "mm/day"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "fmi",
            "sources": {
                "fmi": {"specific_name": "pcp_mm", "unit": "mm/day"},
            },
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "default",
            "sources": {
                "default": {"specific_name": "pet_mm", "unit": "mm/day"},
                "era5_land": {"specific_name": "pe_era5_land", "unit": "mm/day"},
                "fmi": {"specific_name": "pet_fmi", "unit": "mm/day"},
            },
        },
        StandardVariable.SNOW_WATER_EQUIVALENT: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "swe_mm_era5", "unit": "mm"},
                "cci": {"specific_name": "swe_mm_cci3-1", "unit": "mm"},
            },
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "airtemp_C_min", "unit": "°C"},
                "ground_min": {"specific_name": "temperature_gmin", "unit": "°C"},
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "airtemp_C_mean", "unit": "°C"},
            },
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "airtemp_C_max", "unit": "°C"},
            },
        },
        StandardVariable.RELATIVE_HUMIDITY: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "rh_%", "unit": "%"},
            },
        },
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "radiation_global", "unit": "KJ/m^2"},
            },
        },
        StandardVariable.SNOW_DEPTH: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "snowdepth_m", "unit": "cm"},
            },
        },
    }
