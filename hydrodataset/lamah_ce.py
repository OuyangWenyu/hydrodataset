import os
import xarray as xr
from typing import Union, List, Optional

from hydrodataset import HydroDataset, StandardVariable
from tqdm import tqdm
import numpy as np
import pandas as pd
from aqua_fetch import LamaHCE as _AquaFetchLamaHCE
from aqua_fetch.utils import check_attributes


# Define custom LamaHCE class at module level to avoid pickle issues
# Named LamaHCE to maintain compatibility with file naming conventions
class LamaHCE(_AquaFetchLamaHCE):
    """
    Custom LamaHCE class that overrides fetch_static_features to have default value 'all'
    and overrides path properties to adapt to the actual dataset structure.
    """

    @property
    def data_type_dir(self):
        """Override to adapt to the actual LamaH-CE dataset structure.

        The actual structure is:
        lamaHCE/
            2_LamaH-CE_daily/
                A_basins_total_upstrm/
                B_basins_intermediate_all/
                ...
            1_LamaH-CE_daily_hourly/
                A_basins_total_upstrm/
                ...

        Original AquaFetch code expected:
        lamaHCE/
            A_basins_total_upstrm/
            B_basins_intermediate_all/
            ...
        """
        SEP = os.sep

        # Determine which parent folder based on timestep
        if self.timestep == "H":
            parent_folder = "1_LamaH-CE_daily_hourly"
        else:
            parent_folder = "2_LamaH-CE_daily"

        # Find the folder that ends with data_type
        parent_path = os.path.join(self.path, parent_folder)

        # List all directories in parent folder
        if os.path.exists(parent_path):
            dirs = [f for f in os.listdir(parent_path) if f.endswith(self.data_type)]
            if dirs:
                f = dirs[0]
                return os.path.join(parent_path, f)

        # Fallback: try original behavior if new structure doesn't exist
        dirs = [f for f in os.listdir(self.path) if f.endswith(self.data_type)]
        if dirs:
            f = dirs[0]
            return os.path.join(self.path, f)

        raise FileNotFoundError(
            f"Could not find directory ending with '{self.data_type}' "
            f"in {self.path} or {parent_path}"
        )

    @property
    def q_dir(self):
        """Override to adapt to the actual dataset structure."""
        SEP = os.sep

        # Determine which parent folder based on timestep
        if self.timestep == "H":
            parent_folder = "1_LamaH-CE_daily_hourly"
        else:
            parent_folder = "2_LamaH-CE_daily"

        # Try new structure first
        new_path = os.path.join(self.path, parent_folder, "D_gauges", "2_timeseries")
        if os.path.exists(new_path):
            return new_path

        # Fallback to original structure
        return os.path.join(self.path, "D_gauges", "2_timeseries")

    def gauge_attributes(self) -> pd.DataFrame:
        """Override to adapt to the actual dataset structure.

        Original code expected:
        lamaHCE/D_gauges/1_attributes/Gauge_attributes.csv

        Actual structure:
        lamaHCE/2_LamaH-CE_daily/D_gauges/1_attributes/Gauge_attributes.csv
        """
        # Determine which parent folder based on timestep
        if self.timestep == "H":
            parent_folder = "1_LamaH-CE_daily_hourly"
        else:
            parent_folder = "2_LamaH-CE_daily"

        # Try new structure first
        fname = os.path.join(
            self.path, parent_folder, "D_gauges", "1_attributes", "Gauge_attributes.csv"
        )

        if not os.path.exists(fname):
            # Fallback to original structure
            fname = os.path.join(
                self.path, "D_gauges", "1_attributes", "Gauge_attributes.csv"
            )

        df = pd.read_csv(fname, sep=";", index_col="ID")
        df.index = df.index.astype(str)
        return df

    def fetch_static_features(
        self,
        stations: Union[str, List[str]] = "all",
        static_features: Union[str, List[str]] = "all",  # Changed from None to 'all'
    ) -> pd.DataFrame:
        """
        static features of LamaHCE

        Modified to have default static_features='all' instead of None

        Parameters
        ----------
            stations : str
                name/id of station of which to extract the data
            static_features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Examples
        --------
            >>> from aqua_fetch import LamaHCE
            >>> dataset = LamaHCE(timestep='D', data_type='total_upstrm')
            >>> df = dataset.fetch_static_features('99')  # (1, 61)
            ...  # get list of all static features
            >>> dataset.static_features
            >>> dataset.fetch_static_features('99',
            >>> static_features=['area_calc', 'elev_mean', 'agr_fra', 'sand_fra'])  # (1, 4)
        """

        df = self.static_data()

        static_features = check_attributes(
            static_features, self.static_features, "static features"
        )
        stations = check_attributes(stations, self.stations(), "stations")

        df = df[static_features]

        df.index = df.index.astype(str)
        df = df.loc[stations]
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df).transpose()

        return df


class LamahCe(HydroDataset):
    """LamaHCE dataset class extending HydroDataset.

    This class provides access to the LamaHCE dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    def __init__(
        self,
        data_path: str,
        region: Optional[str] = None,
        download: bool = False,
        cache_path: Optional[str] = None,
    ) -> None:
        """Initialize LamaHCE dataset.

        Args:
            data_path: Path to the LamaHCE data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
            cache_path: Path to the cache directory
        """
        super().__init__(data_path, cache_path=cache_path)
        self.region = region
        self.download = download
        # Use the custom LamaHCE class defined at module level
        self.aqua_fetch = LamaHCE(data_path)

    @property
    def _attributes_cache_filename(self):
        return "lamahce_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "lamahce_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1981-01-01", "2019-12-31"]

    # get the information of features from table 3 in "https://doi.org/10.5194/essd-13-4529-2021"
    # Static variable definitions based on inspected data
    _subclass_static_definitions = {
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
    }

    # Dynamic variable mapping based on inspected data
    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "q_cms_obs", "unit": "m^3/s"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "pcp_mm", "unit": "mm"},
            },
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "airtemp_c_max", "unit": "°C"},
                "dp": {"specific_name": "dptemp_c_max_2m", "unit": "°C"},
            },
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "airtemp_c_min", "unit": "°C"},
                "dp": {"specific_name": "dptemp_c_min_2m", "unit": "°C"},
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "airtemp_c_mean", "unit": "°C"},
                "dp": {"specific_name": "dptemp_c_mean_2m", "unit": "°C"},
            },
        },
        StandardVariable.EVAPOTRANSPIRATION: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "total_et", "unit": "mm"},
            },
        },
        StandardVariable.SNOW_WATER_EQUIVALENT: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "swe_mm", "unit": "mm"},
            },
        },
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "solrad_wm2", "unit": "W/m^2"},
            },
        },
        StandardVariable.SOLAR_RADIATION_MAX: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "solrad_wm2_max", "unit": "W/m^2"},
            },
        },
        StandardVariable.THERMAL_RADIATION: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "thermrad_wm2", "unit": "W/m^2"},
            },
        },
        StandardVariable.THERMAL_RADIATION_MAX: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "thermrad_wm2_max", "unit": "W/m^2"},
            },
        },
        StandardVariable.SURFACE_PRESSURE: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "airpres_hpa", "unit": "Pa"},
            },
        },
        StandardVariable.U_WIND_SPEED: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "windspeedu_mps", "unit": "m/s"},
            },
        },
        StandardVariable.V_WIND_SPEED: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "windspeedv_mps", "unit": "m/s"},
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER1: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "volsw_123", "unit": "m^3/m^3"},
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER4: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "volsw_4", "unit": "m^3/m^3"},
            },
        },
    }
