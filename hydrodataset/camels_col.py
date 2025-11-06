"""
Author: Yimeng Zhang
Date: 2025-10-19 19:40:08
LastEditTime: 2025-10-19 19:40:19
LastEditors: Wenyu Ouyang
Description: CAMELS_COL dataset class
FilePath: \hydrodataset\hydrodataset\camels_col.py
Copyright (c) 2021-2026 Wenyu Ouyang. All rights reserved.
"""

from typing import Optional

import numpy as np
import xarray as xr

from aqua_fetch import CAMELS_COL
from hydrodataset import HydroDataset, StandardVariable
from hydroutils import hydro_file


class CamelsCol(HydroDataset):
    """CAMELS_COL dataset class extending RainfallRunoff.

    This class provides access to the CAMELS_COL dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    def __init__(
        self, data_path: str, region: Optional[str] = None, download: bool = False
    ) -> None:
        """Initialize CAMELS_COL dataset.

        Args:
            data_path: Path to the CAMELS_COL data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
        """
        super().__init__(data_path)
        self.region = region
        self.download = download
        self.aqua_fetch = CAMELS_COL(data_path)

    @property
    def _attributes_cache_filename(self):
        return "camels_col_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camels_col_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1981-05-21", "2022-12-31"]

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

    # get the information of features from dataset file "00_CAMELS-COL  Description"
    _subclass_static_definitions = {
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
    }
    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "q_cms_obs", "unit": "m^3/s"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "pcp_mm", "unit": "mm/day"},
            },
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "airtemp_c_max", "unit": "°C"}
            },
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "airtemp_c_min", "unit": "°C"},
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "airtemp_c_mean", "unit": "°C"},
            },
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "pet_mm", "unit": "mm/day"},
            },
        },
    }
