from hydrodataset import HydroDataset, StandardVariable
from aqua_fetch import CAMELS_SE


class CamelsSe(HydroDataset):
    """CAMELS_SE dataset class extending RainfallRunoff.

    This class provides access to the CAMELS_SE dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    def __init__(self, data_path, region=None, download=False):
        """Initialize CAMELS_SE dataset.

        Args:
            data_path: Path to the CAMELS_SE data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
        """
        super().__init__(data_path)
        self.region = region
        self.download = download
        self.aqua_fetch = CAMELS_SE(data_path)

    @property
    def _attributes_cache_filename(self):
        return "camels_se_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camels_se_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1961-01-01", "2020-12-31"]

    # get the information of features from dataset file"Documentation_2024-01-02.pdf"
    _subclass_static_definitions = {
        "p_mean": {"specific_name": "pmean_mm_year", "unit": "mm/year"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
    }

    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "obs_cms",
            "sources": {
                "obs_cms": {"specific_name": "q_cms_obs", "unit": "m^3/s"},
                "obs_mm": {"specific_name": "q_mm_obs", "unit": "mm/day"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "default",
            "sources": {
                "default": {"specific_name": "pcp_mm", "unit": "mm/day"},
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "default",
            "sources": {
                "default": {"specific_name": "airtemp_C_mean", "unit": "°C"},
            },
        },
    }
