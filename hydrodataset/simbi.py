from aqua_fetch import Simbi
from hydrodataset import HydroDataset, StandardVariable


class simbi(HydroDataset):
    """simbi dataset class extending RainfallRunoff.

    This class provides access to the simbi dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    def __init__(self, data_path, region=None, download=False):
        """Initialize simbi dataset.

        Args:
            data_path: Path to the simbi data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
            cache_path: Path to the cache directory
        """
        super().__init__(data_path)
        self.region = region
        self.download = download
        self.aqua_fetch = Simbi(data_path)

    @property
    def _attributes_cache_filename(self):
        return "simbi_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "simbi_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1920-01-01", "2005-12-31"]

    # get the information of features from dataset file "SIMBI_README"
    _subclass_static_definitions = {
        "area": {"specific_name": "area_km2", "unit": "km^2"},
        "p_mean": {"specific_name": "p_mon_avg", "unit": "mm/month"},
    }

    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "q_cms_obs", "unit": "m^3/s"}
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "observations",
            "sources": {"observations": {"specific_name": "pcp_mm", "unit": "mm/day"}},
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "airtemp_c_mean", "unit": "°C"}
            },
        },
    }
