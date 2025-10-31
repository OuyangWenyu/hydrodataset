"""
Author: Wenyu Ouyang
Date: 2025-10-31
LastEditTime: 2025-10-31
LastEditors: Wenyu Ouyang
Description: CAMELS-IND dataset class.
FilePath: \hydrodataset\hydrodataset\camels_ind.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import os
from pathlib import Path

from hydrodataset import HydroDataset, StandardVariable
from aqua_fetch import CAMELS_IND
from hydroutils import hydro_file
from tqdm import tqdm


class CustomCAMELS_IND(CAMELS_IND):
    """Custom CAMELS_IND class that supports the latest dataset version.

    This class overrides the default CAMELS_IND implementation to support
    the new file structure and naming conventions in the latest dataset version.
    Defined at module level to ensure proper pickle serialization for multiprocessing.
    """

    url = "https://zenodo.org/records/14999580"

    def __init__(self, data_path):
        """Custom initialization that uses the new file structure and names."""
        import pandas as pd

        # Don't call parent __init__ to avoid reading old file names
        # Manually set necessary attributes required by RainfallRunoff
        self.path = data_path
        self._actual_data_path = data_path
        self.name = "CAMELS_IND"
        self.timestep = "D"  # Daily timestep
        self.processes = None  # Use default multiprocessing
        self.verbosity = 0  # No verbose output
        self.to_netcdf = False  # Don't auto-convert to netCDF

        # Determine the CAMELS_IND directory path for reading files
        if os.path.basename(data_path).upper() == "CAMELS_IND":
            camels_ind_dir = data_path
        else:
            # data_path is the parent directory, try both uppercase and lowercase
            possible_paths = [
                os.path.join(data_path, "CAMELS_IND"),
                os.path.join(data_path, "camels_ind"),
            ]
            camels_ind_dir = None
            for p in possible_paths:
                if os.path.exists(p):
                    camels_ind_dir = p
                    break
            if camels_ind_dir is None:
                # Default to uppercase if directory doesn't exist yet
                camels_ind_dir = os.path.join(data_path, "CAMELS_IND")

        # Read station names from the updated file (using gauge_id)
        names_file = os.path.join(
            camels_ind_dir,
            "CAMELS_IND_All_Catchments",
            "attributes_txt",
            "camels_ind_name.txt",
        )
        try:
            # Read with semicolon separator and use first column (gauge_id) as index
            names = pd.read_csv(names_file, sep=";", index_col=0, dtype={0: str})
            # Get gauge_id from index
            id_str = names.index.to_list()
            id_int = names.index.astype(int).to_list()
            self.id_map = {str(k): v for k, v in zip(id_int, id_str)}
            self._stations = id_str

            # Initialize _static_features and _dynamic_features
            try:
                self._static_features = self._static_data().columns.to_list()
            except Exception:
                self._static_features = []

            try:
                if self._stations:
                    self._dynamic_features = self._read_stn_dyn(
                        self.stations()[0]
                    ).columns.to_list()
                else:
                    self._dynamic_features = []
            except Exception:
                self._dynamic_features = []

        except FileNotFoundError:
            print(f"Warning: Could not find {names_file}, station list may be empty")
            self._stations = []
            self._static_features = []
            self._dynamic_features = []
        except Exception as e:
            print(f"Error reading stations from {names_file}: {e}")
            self._stations = []
            self._static_features = []
            self._dynamic_features = []

    def _get_camels_ind_dir(self):
        """Helper to determine the CAMELS_IND directory path."""
        if hasattr(self, "_actual_data_path"):
            data_path = self._actual_data_path
            if os.path.basename(data_path).upper() == "CAMELS_IND":
                return data_path
            # Try both uppercase and lowercase
            for dirname in ["CAMELS_IND", "camels_ind"]:
                candidate = os.path.join(data_path, dirname)
                if os.path.exists(candidate):
                    return candidate
            return os.path.join(data_path, "CAMELS_IND")
        # Fallback to self.path if _actual_data_path not set
        return self.path

    @property
    def static_path(self):
        """Return the path to static attributes directory."""
        camels_ind_dir = self._get_camels_ind_dir()
        return os.path.join(
            camels_ind_dir, "CAMELS_IND_All_Catchments", "attributes_txt"
        )

    @property
    def q_path(self):
        """Return the path to streamflow timeseries directory."""
        camels_ind_dir = self._get_camels_ind_dir()
        return os.path.join(
            camels_ind_dir, "CAMELS_IND_All_Catchments", "streamflow_timeseries"
        )

    @property
    def forcings_path(self):
        """Return the path to forcings directory."""
        camels_ind_dir = self._get_camels_ind_dir()
        return os.path.join(
            camels_ind_dir, "CAMELS_IND_All_Catchments", "catchment_mean_forcings"
        )

    def stn_forcing_path(self, stn: str):
        """Custom forcing path without subdirectory structure."""
        forcings_path = self.forcings_path
        # Get the station ID with leading zeros from id_map
        stn_id = self.id_map.get(stn, stn)
        return os.path.join(forcings_path, f"{stn_id}.csv")

    def stations(self):
        """Return station IDs without leading zeros (to match original implementation)."""
        # Convert to int and back to str to remove leading zeros
        return [str(int(stn)) for stn in self._stations]

    @property
    def static_features(self):
        """Return static features."""
        return self._static_features

    @property
    def dynamic_features(self):
        """Return dynamic features."""
        return self._dynamic_features

    def _maybe_to_netcdf(self, *args, **kwargs):
        """Disable _maybe_to_netcdf to avoid multiprocessing pickle issues."""
        pass


class CamelsInd(HydroDataset):
    """CAMELS_IND dataset class extending HydroDataset.

    This class provides access to the CAMELS_IND dataset, which contains
    hydrological and meteorological data for various watersheds in India.
    It uses a custom implementation to support the latest dataset version.

    The class relies on AquaFetch for data reading but overrides certain
    methods to support the new file structure in the latest Zenodo release.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        aqua_fetch: CustomCAMELS_IND instance for data access
    """

    def __init__(self, data_path, region=None, download=False):
        """Initialize CAMELS_IND dataset.

        Args:
            data_path: Path to the CAMELS_IND data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
        """
        super().__init__(data_path)
        self.region = region
        self.download = download

        try:
            # Use custom class that supports the latest dataset version
            self.aqua_fetch = CustomCAMELS_IND(data_path)
        except Exception as e:
            print(e)
            # If initialization fails, try to extract zip files
            check_zip_extract = False
            zip_files = [
                "CAMELS_IND_All_Catchments.zip",
                "CAMELS_IND_Catchments_Streamflow_Sufficient.zip",
            ]
            for filename in tqdm(zip_files, desc="Checking zip files"):
                extracted_dir = self.data_source_dir.joinpath(
                    "CAMELS_IND", filename[:-4]
                )
                if not extracted_dir.exists():
                    check_zip_extract = True
                    break
            if check_zip_extract:
                hydro_file.zip_extract(self.data_source_dir.joinpath("CAMELS_IND"))
            # Retry initialization after extraction
            self.aqua_fetch = CustomCAMELS_IND(data_path)

    @property
    def _attributes_cache_filename(self):
        return "camels_ind_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camels_ind_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1980-01-01", "2020-12-31"]

    # get the information of features from dataset file"00_CAMELS_IND_Data_Description.pdf"
    _subclass_static_definitions = {
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
    }

    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "obs",
            "sources": {"obs": {"specific_name": "q_cms_obs", "unit": "m^3/s"}},
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "imd",
            "sources": {"imd": {"specific_name": "pcp_mm", "unit": "mm/day"}},
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "imd",
            "sources": {"imd": {"specific_name": "airtemp_c_max", "unit": "°C"}},
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "imd",
            "sources": {"imd": {"specific_name": "airtemp_c_min", "unit": "°C"}},
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "imd",
            "sources": {"imd": {"specific_name": "airtemp_c_mean", "unit": "°C"}},
        },
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "imdaa",
            "sources": {"imdaa": {"specific_name": "solrad_wm2", "unit": "W/m^2"}},
        },
        StandardVariable.LONGWAVE_SOLAR_RADIATION: {
            "default_source": "imdaa",
            "sources": {"imdaa": {"specific_name": "lwdownrad_wm2", "unit": "W/m^2"}},
        },
        StandardVariable.WIND_SPEED: {
            "default_source": "imdaa",
            "sources": {"imdaa": {"specific_name": "windspeed_mps", "unit": "m/s"}},
        },
        StandardVariable.V_WIND_SPEED: {
            "default_source": "imdaa",
            "sources": {"imdaa": {"specific_name": "windspeedv_mps", "unit": "m/s"}},
        },
        StandardVariable.U_WIND_SPEED: {
            "default_source": "imdaa",
            "sources": {"imdaa": {"specific_name": "windspeedu_mps", "unit": "m/s"}},
        },
        StandardVariable.RELATIVE_HUMIDITY: {
            "default_source": "imdaa",
            "sources": {"imdaa": {"specific_name": "rh_", "unit": "%"}},
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "default",
            "sources": {
                "default": {"specific_name": "pet_mm", "unit": "mm/day"},
                "gleam": {"specific_name": "pet_mm_gleam", "unit": "mm/day"},
            },
        },
        StandardVariable.EVAPOTRANSPIRATION: {
            "default_source": "gleam",
            "sources": {"gleam": {"specific_name": "aet_mm_gleam", "unit": "mm/day"}},
        },
        StandardVariable.EVAPORATION: {
            "default_source": "canopy",
            "sources": {
                "canopy": {"specific_name": "evap_canopy", "unit": "mm/day"},
                "surface": {"specific_name": "evap_surface", "unit": "mm/day"},
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER1: {
            "default_source": "imdaa",
            "sources": {"imdaa": {"specific_name": "sm_lvl1", "unit": "kg/m^2"}},
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER2: {
            "default_source": "imdaa",
            "sources": {"imdaa": {"specific_name": "sm_lvl2", "unit": "kg/m^2"}},
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER3: {
            "default_source": "imdaa",
            "sources": {"imdaa": {"specific_name": "sm_lvl3", "unit": "kg/m^2"}},
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER4: {
            "default_source": "imdaa",
            "sources": {"imdaa": {"specific_name": "sm_lvl4", "unit": "kg/m^2"}},
        },
    }
