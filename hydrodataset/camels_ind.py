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
from typing import Optional

import numpy as np
import pandas as pd
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

    def __init__(self, uri):
        """Custom initialization that uses the new file structure and names."""
        import pandas as pd

        # Don't call parent __init__ to avoid reading old file names
        # Manually set necessary attributes required by RainfallRunoff
        self.path = uri
        self._actual_data_path = uri
        self.name = "CAMELS_IND"
        self.timestep = "D"  # Daily timestep
        self.processes = None  # Use default multiprocessing
        self.verbosity = 0  # No verbose output
        self.to_netcdf = False  # Don't auto-convert to netCDF

        # Determine the CAMELS_IND directory path for reading files
        if os.path.basename(uri).upper() == "CAMELS_IND":
            camels_ind_dir = uri
        else:
            # uri is the parent directory, try both uppercase and lowercase
            possible_paths = [
                os.path.join(uri, "CAMELS_IND"),
                os.path.join(uri, "camels_ind"),
            ]
            camels_ind_dir = None
            for p in possible_paths:
                if os.path.exists(p):
                    camels_ind_dir = p
                    break
            if camels_ind_dir is None:
                # Default to uppercase if directory doesn't exist yet
                camels_ind_dir = os.path.join(uri, "CAMELS_IND")

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

    _FORCING_COL_MAP = {
        "prcp(mm/day)":         "pcp_mm",
        "tmax(C)":              "airtemp_c_max",
        "tmin(C)":              "airtemp_c_min",
        "tavg(C)":              "airtemp_c_mean",
        "srad_lw(w/m2)":        "lwdownrad_wm2",
        "srad_sw(w/m2)":        "solrad_wm2",
        "wind_u(m/s)":          "windspeedu_mps",
        "wind_v(m/s)":          "windspeedv_mps",
        "wind(m/s)":            "windspeed_mps",
        "rel_hum(%)":           "rh_",
        "pet(mm/day)":          "pet_mm",
        "pet_gleam(mm/day)":    "pet_mm_gleam",
        "aet_gleam(mm/day)":    "aet_mm_gleam",
        "evap_canopy(mm/day)":  "evap_canopy",
        "evap_surface(mm/day)": "evap_surface",
        "sm_lvl1(kg/m2)":       "sm_lvl1",
        "sm_lvl2(kg/m2)":       "sm_lvl2",
        "sm_lvl3(kg/m2)":       "sm_lvl3",
        "sm_lvl4(kg/m2)":       "sm_lvl4",
    }
    _ATTR_FILES = [
        "camels_ind_topo.txt",
        "camels_ind_clim.txt",
        "camels_ind_geol.txt",
        "camels_ind_hydro.txt",
        "camels_ind_land.txt",
        "camels_ind_soil.txt",
        "camels_ind_anth.txt",
        "camels_ind_name.txt",
    ]
    _DATA_REL = "CAMELS_IND/CAMELS_IND_All_Catchments"
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

    def __init__(
        self, uri: str, region: Optional[str] = None, download: bool = False
    ) -> None:
        """Initialize CAMELS_IND dataset.

        Args:
            uri: Path to the data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
        """
        super().__init__(uri)
        self.region = region
        self.download = download
        if str(uri).startswith("s3://"):
            return

        try:
            # Use custom class that supports the latest dataset version
            self.aqua_fetch = CustomCAMELS_IND(uri)
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
            self.aqua_fetch = CustomCAMELS_IND(uri)

    def read_object_ids(self) -> np.ndarray:
        uri = str(self.data_source_dir).rstrip("/")
        forcing_rel = f"{self._DATA_REL}/catchment_mean_forcings"
        if self._is_cloud():
            fs = self._make_s3fs()
            names = [p.split("/")[-1] for p in fs.ls(f"{uri}/{forcing_rel}".removeprefix("s3://"))]
        else:
            names = os.listdir(os.path.join(uri, *forcing_rel.split("/")))
        ids = sorted(n.replace(".csv", "") for n in names if n.endswith(".csv"))
        return np.array(ids)

    def cache_attributes_to_zarr(self) -> None:
        import zarr
        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")
        attr_base = f"{uri}/{self._DATA_REL}/attributes_txt"
        dfs = []
        for fname in self._ATTR_FILES:
            path = f"{attr_base}/{fname}".removeprefix("s3://")
            try:
                with fs.open(path) as fh:
                    df = pd.read_csv(fh, sep=";", index_col="gauge_id",
                                     dtype={"gauge_id": str})
                df.index = df.index.astype(str)
                dfs.append(df)
            except Exception as e:
                print(f"  WARN {fname}: {e}")
        static = pd.concat(dfs, axis=1)
        static = static.loc[~static.index.duplicated(keep="first")]
        stations = self.read_object_ids().tolist()
        static = static.reindex(stations)
        static.columns = self._clean_feature_names(list(static.columns))
        static = static.rename(columns={"cwc_area": "area_km2", "cwc_lat": "lat"})

        zarr_name = self._attributes_cache_filename.replace(".nc", ".zarr")
        out, opts = self._zarr_path_and_opts(zarr_name)
        n = len(stations)
        root = zarr.open_group(out, mode="w", storage_options=opts, zarr_format=2)
        for col in static.columns:
            vals = static[col].values.astype(str) if static[col].dtype == object else static[col].values
            arr = root.create_array(col, shape=(n,), chunks=(n,), dtype=vals.dtype)
            arr[:] = vals
            arr.attrs["_ARRAY_DIMENSIONS"] = ["basin"]
        basin_arr = root.create_array("basin", shape=(n,), chunks=(n,), dtype=str)
        basin_arr[:] = stations
        basin_arr.attrs["_ARRAY_DIMENSIONS"] = ["basin"]
        root.attrs["coordinates"] = "basin"
        print(f"Attributes zarr written to: {out}")

    def cache_timeseries_to_zarr(self) -> None:
        import zarr
        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")
        base = f"{uri}/{self._DATA_REL}"

        stations = self.read_object_ids().tolist()
        all_times = pd.date_range(self.default_t_range[0], self.default_t_range[1], freq="D")
        n, nt = len(stations), len(all_times)
        times_ns = all_times.asi8
        all_vars = list(self._FORCING_COL_MAP.values()) + ["q_cms_obs"]

        # Read wide streamflow once
        print("Reading streamflow_observed.csv...")
        q_df = None
        try:
            q_path = f"{base}/streamflow_timeseries/streamflow_observed.csv".removeprefix("s3://")
            with fs.open(q_path) as fh:
                q_raw = pd.read_csv(fh)
            q_raw.index = pd.to_datetime(
                {"year": q_raw["year"], "month": q_raw["month"], "day": q_raw["day"]}
            )
            q_raw = q_raw.drop(columns=["year", "month", "day"])
            q_raw.columns = q_raw.columns.astype(str)
            q_df = q_raw.reindex(all_times)
        except Exception as e:
            print(f"  WARN streamflow: {e}")

        data: dict[str, np.ndarray] = {vn: np.full((n, nt), np.nan) for vn in all_vars}
        print(f"Reading {n} station forcings from OSS...")
        for i, stn in enumerate(tqdm(stations, desc="CAMELS_IND zarr")):
            forcing_path = f"{base}/catchment_mean_forcings/{stn}.csv".removeprefix("s3://")
            try:
                with fs.open(forcing_path) as fh:
                    df = pd.read_csv(fh)
                df.index = pd.to_datetime(
                    {"year": df["year"], "month": df["month"], "day": df["day"]}
                )
                df = df.reindex(all_times)
                for raw_col, zarr_vn in self._FORCING_COL_MAP.items():
                    if raw_col in df.columns:
                        data[zarr_vn][i] = df[raw_col].values.astype(float)
            except Exception as e:
                print(f"  WARN {stn}: {e}")

            # Streamflow: column is int(stn) without leading zeros
            if q_df is not None:
                q_col = str(int(stn))
                if q_col in q_df.columns:
                    data["q_cms_obs"][i] = q_df[q_col].values.astype(float)

        zarr_name = self._timeseries_cache_filename.replace(".nc", ".zarr")
        out, opts = self._zarr_path_and_opts(zarr_name)
        root = zarr.open_group(out, mode="w", storage_options=opts, zarr_format=2)
        for vn in all_vars:
            arr = root.create_array(vn, shape=(n, nt), chunks=(min(n, 100), min(nt, 365)), dtype="float64")
            arr[:] = data[vn]
            arr.attrs["_ARRAY_DIMENSIONS"] = ["basin", "time"]

        time_arr = root.create_array("time", shape=(nt,), chunks=(min(nt, 365),), dtype="int64")
        time_arr[:] = times_ns
        time_arr.attrs["_ARRAY_DIMENSIONS"] = ["time"]
        time_arr.attrs["units"] = "nanoseconds since 1970-01-01"
        time_arr.attrs["calendar"] = "proleptic_gregorian"

        basin_arr = root.create_array("basin", shape=(n,), chunks=(n,), dtype=str)
        basin_arr[:] = stations
        basin_arr.attrs["_ARRAY_DIMENSIONS"] = ["basin"]

        root.attrs["coordinates"] = "basin time"
        print(f"Timeseries zarr written to: {out}")

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
            "sources": {"imd": {"specific_name": "airtemp_c_max", "unit": "掳C"}},
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "imd",
            "sources": {"imd": {"specific_name": "airtemp_c_min", "unit": "掳C"}},
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "imd",
            "sources": {"imd": {"specific_name": "airtemp_c_mean", "unit": "掳C"}},
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
