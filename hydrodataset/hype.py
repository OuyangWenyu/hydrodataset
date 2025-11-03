from aqua_fetch import HYPE
from hydrodataset import HydroDataset, StandardVariable


class Hype(HydroDataset):
    """HYPE dataset class extending RainfallRunoff.

    This class provides access to the HYPE dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    def __init__(self, data_path, region=None, download=False, time_step="daily"):
        """Initialize HYPE dataset.

        Args:
            data_path: Path to the HYPE data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
            time_step: Time step for the data ('daily', 'month', or 'year', default: 'daily')
            cache_path: Path to the cache directory
        """
        super().__init__(data_path)
        self.region = region
        self.download = download
        self.time_step = time_step
        self.aqua_fetch = HYPE(time_step, data_path)

    @property
    def _attributes_cache_filename(self):
        return "hype_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "hype_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1985-01-01", "2019-12-31"]

    # HYPE dataset does not have static attributes from aqua_fetch
    # We will need to compute area and p_mean from other sources or leave them empty
    _subclass_static_definitions = {
        # "area": {"specific_name": "area_km2", "unit": "km^2"},  # Not available
        # "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},  # Can be computed from prec_mm
    }

    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "streamflow_mm", "unit": "mm/day"}
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "observations",
            "sources": {"observations": {"specific_name": "prec_mm", "unit": "mm/day"}},
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "observations",
            "sources": {"observations": {"specific_name": "pet_mm", "unit": "mm/day"}},
        },
        StandardVariable.EVAPOTRANSPIRATION: {
            "default_source": "observations",
            "sources": {"observations": {"specific_name": "aet_mm", "unit": "mm/day"}},
        },
        StandardVariable.SOIL_MOISTURE: {
            "default_source": "observations",
            "sources": {"observations": {"specific_name": "sm_mm", "unit": "mm"}},
        },
    }
