import os
from typing import Optional

from hydrodataset import HydroDataset, StandardVariable
from tqdm import tqdm
from aqua_fetch import CAMELS_NZ as _AquaFetchCAMELS_NZ


# Define the custom CAMELS_NZ class at module level to avoid pickle errors
# Named CAMELS_NZ to maintain compatibility with file naming conventions
class CAMELS_NZ(_AquaFetchCAMELS_NZ):
    """Custom CAMELS_NZ class with updated URLs and timestep support.

    This class extends the base CAMELS_NZ to support the newer dataset version
    and provides flexible timestep configuration (hourly or daily).

    Attributes:
        timestep: Time step for the data ('H' for hourly, 'D' for daily)
    """

    # Override the base URL for the new dataset version
    url = "https://figshare.canterbury.ac.nz/ndownloader/articles/28827644/versions/2"

    def __init__(self, data_path, timestep="H", **kwargs):
        """Initialize CustomCamelsNz.

        Args:
            data_path: Path to the data directory
            timestep: Time step for the data ('H' for hourly, 'D' for daily)
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(data_path, **kwargs)
        self.timestep = timestep

    @property
    def temp_path(self):
        folder_name = (
            "CAMELS_NZ_hourly_Temperature"
            if self.timestep == "H"
            else "CAMELS_NZ_daily_Temperature"
        )
        return os.path.join(self.path, "camels_nz", folder_name)

    @property
    def precip_path(self):
        folder_name = (
            "CAMELS_NZ_hourly_Precipitation"
            if self.timestep == "H"
            else "CAMELS_NZ_daily_Precipitation"
        )
        return os.path.join(self.path, "camels_nz", folder_name)

    @property
    def q_path(self):
        folder_name = (
            "CAMELS_NZ_hourly_Streamflow"
            if self.timestep == "H"
            else "CAMELS_NZ_daily_Streamflow"
        )
        return os.path.join(self.path, "camels_nz", folder_name)

    @property
    def shapefile_path(self):
        return os.path.join(self.path, "camels_nz", "CAMELS_NZ_Shapefiles")

    @property
    def pet_path(self):
        folder_name = (
            "CAMELS_NZ_hourly_PET" if self.timestep == "H" else "CAMELS_NZ_daily_PET"
        )
        return os.path.join(self.path, "camels_nz", folder_name)

    @property
    def rh_path(self):
        folder_name = (
            "CAMELS_NZ_hourly_Relative_Humidity"
            if self.timestep == "H"
            else "CAMELS_NZ_daily_Relative_Humidity"
        )
        return os.path.join(self.path, "camels_nz", folder_name)

    def _read_stn_dyn_para(self, stn: str, para_name: str):
        """Override _read_stn_dyn_para to handle timestep-dependent file names.

        Args:
            stn: Station ID
            para_name: Parameter name to read

        Returns:
            pandas.Series: Time series data for the station and parameter
        """
        import pandas as pd
        import numpy as np

        stn_q = pd.Series(dtype=np.float32, name=stn)

        fname = {"Relative_humidity": "RH"}

        # Construct file name based on timestep
        prefix = fname.get(para_name, para_name)
        if self.timestep == "D":
            prefix = f"daily_{prefix}"

        fpath = os.path.join(
            self._path_map[para_name], f"{prefix}_station_id_{stn}.csv"
        )

        if os.path.exists(fpath):
            if para_name == "flow" and stn in self._nodata_stns:
                return stn_q

            try:
                stn_q = pd.read_csv(
                    fpath, index_col=0, parse_dates=True, na_values=["NA  "]
                )
            except pd.errors.EmptyDataError:
                warning_prefix = "daily_" if self.timestep == "D" else ""
                print(
                    f"Warning: {warning_prefix}{para_name}_station_id_{stn}.csv is empty. Skipping station {stn}."
                )
                return stn_q

            format = "%m/%d/%Y %H:%M" if self.timestep == "H" else "%m/%d/%Y"
            if para_name == "flow" and stn == "57521" and self.timestep == "H":
                format = "%d/%m/%Y %H:%M"
            elif para_name == "flow" and stn == "57521" and self.timestep == "D":
                format = "%d/%m/%Y"

            stn_q.index = pd.to_datetime(stn_q.index, format=format)
            stn_q = stn_q[para_name].astype(np.float32).rename(stn)
        else:
            if self.verbosity > 1:
                warning_prefix = "daily_" if self.timestep == "D" else ""
                print(
                    f"Warning: {warning_prefix}{para_name}_station_id_{stn}.csv does not exist. Skipping station {stn}."
                )
            stn_q = pd.Series(dtype=np.float32, name=stn)

        # Remove rows with duplicated index
        stn_q = stn_q[~stn_q.index.duplicated(keep="first")]

        return stn_q

    def _maybe_to_netcdf(self, *args, **kwargs):
        """Override to disable netcdf conversion."""
        pass


class CamelsNz(HydroDataset):
    """CAMELS_NZ dataset class.

    This class uses a custom data reading implementation to support a newer
    dataset version than the one supported by the underlying aquafetch library.
    It overrides the download URLs and provides its own parsing and caching logic.

    The dataset supports both hourly ('H') and daily ('D') timesteps.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        timestep: Time step for the data ('H' for hourly, 'D' for daily)
    """

    def __init__(
        self,
        data_path: str,
        region: Optional[str] = None,
        download: bool = False,
        timestep: str = "H",
    ) -> None:
        """Initialize CAMELS_NZ dataset.

        Args:
            data_path: Path to the CAMELS_NZ data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
            timestep: Time step for the data ('H' for hourly, 'D' for daily, default: 'H')
        """
        super().__init__(data_path)
        self.region = "NZ" if region is None else region
        self.download = download
        self.timestep = timestep

        # Instantiate our custom CAMELS_NZ class with timestep support
        self.aqua_fetch = CAMELS_NZ(data_path, timestep=timestep)

    @property
    def _attributes_cache_filename(self):
        return f"camels_nz_{self.timestep.lower()}_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return f"camels_nz_{self.timestep.lower()}_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1972-01-01", "2024-08-02"]

    # Static variable definitions for CAMELS-NZ
    # Note: specific_name should be the cleaned version (lowercase, no spaces)
    # as stored in the cache file after _clean_feature_names() processing
    _subclass_static_definitions = {
        "area": {"specific_name": "area_km2", "unit": "km^2"},
        "p_mean": {"specific_name": "meanannualrainfall", "unit": "mm"},
    }

    # Dynamic variable mapping for CAMELS-NZ
    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "obs",
            "sources": {"obs": {"specific_name": "q_cms_obs", "unit": "m^3/s"}},
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "default",
            "sources": {"default": {"specific_name": "pcp_mm", "unit": "mm/day"}},
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "default",
            "sources": {"default": {"specific_name": "airtemp_c_mean", "unit": "°C"}},
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "default",
            "sources": {"default": {"specific_name": "pet_mm", "unit": "mm/day"}},
        },
        StandardVariable.RELATIVE_HUMIDITY: {
            "default_source": "default",
            "sources": {"default": {"specific_name": "rh_", "unit": "%"}},
        },
    }
