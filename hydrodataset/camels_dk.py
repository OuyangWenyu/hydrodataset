"""
Author: Yimeng Zhang
Date: 2025-10-19 19:40:08
LastEditTime: 2025-10-19 19:54:34
LastEditors: Wenyu Ouyang
Description: CAMELS_DK dataset class
FilePath: \hydrodataset\hydrodataset\camels_dk.py
Copyright (c) 2021-2026 Wenyu Ouyang. All rights reserved.
"""

from typing import Optional

from hydrodataset import HydroDataset, StandardVariable
from aqua_fetch import CAMELS_DK


class CamelsDk(HydroDataset):
    """CAMELS_DK dataset class extending RainfallRunoff.

    This class provides access to the CAMELS_DK dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    def __init__(
        self, data_path: str, region: Optional[str] = None, download: bool = False
    ) -> None:
        """Initialize CAMELS_DK dataset.

        Args:
            data_path: Path to the CAMELS_DK data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
        """
        super().__init__(data_path)
        self.region = region
        self.download = download
        self.aqua_fetch = CAMELS_DK(data_path)

    @property
    def _attributes_cache_filename(self):
        return "camels_dk_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camels_dk_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1989-01-02", "2023-12-31"]

    # get the information of features from dataset file"Data_description.pdf"
    _subclass_static_definitions = {
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
        "gauge_lat": {"specific_name": "lat", "unit": "degree"},
        "gauge_lon": {"specific_name": "long", "unit": "degree"},
        "elev_mean": {"specific_name": "dem_mean", "unit": "m"},
        "pet_mean": {"specific_name": "pet_mean", "unit": "mm/day"},
    }

    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "dkmodel",
            "sources": {
                "dkmodel": {"specific_name": "q_cms_obs", "unit": "m^3/s"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "dmi",
            "sources": {
                "dmi": {"specific_name": "pcp_mm", "unit": "mm/day"},
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "dmi",
            "sources": {
                "dmi": {"specific_name": "airtemp_c_mean", "unit": "°C"},
            },
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "dmi",
            "sources": {
                "dmi": {"specific_name": "pet_mm", "unit": "mm/day"},
            },
        },
        StandardVariable.EVAPOTRANSPIRATION: {
            "default_source": "dkm_model",
            "sources": {
                "dkm_model": {"specific_name": "aet_mm", "unit": "mm/day"},
            },
        },
    }
