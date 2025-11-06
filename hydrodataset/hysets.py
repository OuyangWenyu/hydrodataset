from typing import Optional

import numpy as np
import xarray as xr

from aqua_fetch import HYSETS
from hydrodataset import HydroDataset, StandardVariable


class Hysets(HydroDataset):
    """HYsets dataset class extending RainfallRunoff.

    This class provides access to the HYsets dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    def __init__(
        self, data_path: str, region: Optional[str] = None, download: bool = False
    ) -> None:
        """Initialize HYsets dataset.

        Args:
            data_path: Path to the HYsets data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
        """
        super().__init__(data_path)
        self.region = region
        self.download = download
        self.aqua_fetch = HYSETS(data_path)

    @property
    def _attributes_cache_filename(self):
        return "hysets_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "hysets_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1950-01-01", "2023-12-31"]

    def cache_attributes_xrdataset(self):
        """Override base method to add calculated p_mean from precipitation timeseries.

        This method:
        1. Calls parent method to create base attribute cache
        2. Reads precipitation timeseries data
        3. Calculates mean precipitation (p_mean) for each basin
        4. Adds p_mean to the attribute dataset
        5. Saves the updated cache
        """
        # Step 1: Create base attribute cache using parent method
        print("Creating base attribute cache...")
        super().cache_attributes_xrdataset()

        # Step 2: Load the base cache file
        cache_file = self.cache_dir.joinpath(self._attributes_cache_filename)
        with xr.open_dataset(cache_file) as ds_attr:
            ds_attr = ds_attr.load()  # Load into memory

        print("Calculating p_mean from precipitation timeseries...")

        # Step 3: Read precipitation timeseries for all basins
        basin_ids = self.read_object_ids().tolist()

        try:
            # Read full precipitation timeseries
            prcp_ts = self.read_ts_xrdataset(
                gage_id_lst=basin_ids,
                t_range=self.default_t_range,
                var_lst=["precipitation"],
            )

            # Step 4: Calculate temporal mean for each basin
            # The result is a DataArray with dimension (basin,)
            p_mean_values = prcp_ts["precipitation"].mean(dim="time")

            # Add units attribute
            p_mean_values.attrs["units"] = "mm/day"
            p_mean_values.attrs["description"] = (
                "Mean daily precipitation (calculated from timeseries)"
            )

            # Step 5: Add p_mean to the attribute dataset
            ds_attr["p_mean"] = p_mean_values

            print(f"Successfully calculated p_mean for {len(basin_ids)} basins")

        except Exception as e:
            print(f"Warning: Could not calculate p_mean from precipitation data: {e}")
            print("Creating p_mean with NaN values as placeholder")
            # Create p_mean with NaN values if calculation fails
            p_mean_nan = xr.DataArray(
                np.full(len(basin_ids), np.nan),
                coords={"basin": basin_ids},
                dims=["basin"],
                attrs={
                    "units": "mm/day",
                    "description": "Mean daily precipitation (not available)",
                },
            )
            ds_attr["p_mean"] = p_mean_nan

        # Step 6: Save the updated cache file
        print(f"Saving updated attribute cache with p_mean to: {cache_file}")
        ds_attr.to_netcdf(cache_file, mode="w")
        print("Successfully saved attribute cache with p_mean")

    _subclass_static_definitions = {
        "area": {"specific_name": "area_km2", "unit": "km^2"},
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
    }

    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "observations_cms",
            "sources": {
                "observations_cms": {"specific_name": "q_cms_obs", "unit": "m^3/s"},
                "observations_mm": {"specific_name": "q_mm_obs", "unit": "mm/day"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "observations",
            "sources": {"observations": {"specific_name": "pcp_mm", "unit": "mm/day"}},
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "airtemp_c_2m_max", "unit": "°C"}
            },
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "airtemp_c_2m_min", "unit": "°C"}
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "dptemp_c_mean_2m", "unit": "°C"}
            },
        },
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "solrad_wm2", "unit": "W/m^2"},
                "net": {"specific_name": "solradnet_wm2", "unit": "W/m^2"},
            },
        },
        StandardVariable.EVAPORATION: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "evap_mm", "unit": "mm/day"},
                "snow": {"specific_name": "evap_mm_snow", "unit": "mm/day"},
            },
        },
        StandardVariable.SNOW_WATER_EQUIVALENT: {
            "default_source": "observations",
            "sources": {"observations": {"specific_name": "swe_mm", "unit": "mm"}},
        },
        StandardVariable.SURFACE_PRESSURE: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "airpres_hpa", "unit": "hPa"}
            },
        },
        StandardVariable.U_WIND_SPEED: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "windspeedu_mps", "unit": "m/s"}
            },
        },
        StandardVariable.V_WIND_SPEED: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "windspeedv_mps", "unit": "m/s"}
            },
        },
        StandardVariable.LONGWAVE_SOLAR_RADIATION: {
            "default_source": "downward",
            "sources": {
                "downward": {"specific_name": "lwdownrad_wm2", "unit": "W/m^2"},
                "net": {"specific_name": "lwnetrad_wm2", "unit": "W/m^2"},
            },
        },
        StandardVariable.SNOW_DENSITY: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "snowdensity_kgm3", "unit": "kg/m^3"}
            },
        },
    }
