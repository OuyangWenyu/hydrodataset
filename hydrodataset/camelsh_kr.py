import os
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from hydrodataset import HydroDataset, StandardVariable
from tqdm import tqdm
from aqua_fetch import CAMELS_SK


class CamelshKr(HydroDataset):

    # raw CSV column → zarr variable name (same name kept where already matching NC)
    _COL_MAP = {
        "total_precipitation":         "total_precipitation",
        "temperature_2m":              "temperature_2m",
        "dewpoint_temperature_2m":     "dewpoint_temperature_2m",
        "snow_cover":                  "snow_cover",
        "snow_depth":                  "snow_depth",
        "potential_evaporation":       "potential_evaporation",
        "u_component_of_wind_10m":     "u_component_of_wind_10m",
        "v_component_of_wind_10m":     "v_component_of_wind_10m",
        "surface_pressure":            "surface_pressure",
        "surface_net_thermal_radiation": "surface_net_thermal_radiation",
        "surface_net_solar_radiation": "surface_net_solar_radiation",
        "precip_obs":                  "precip_obs",
        "air_temp_obs":                "air_temp_obs",
        "wind_dir_obs":                "wind_dir_obs",
        "wind_sp_obs":                 "wind_sp_obs",
        "streamflow":                  "q_cms_obs",
        "water_level":                 "water_level",
    }
    _ATTR_FILES = [
        "attributes_general.csv",
        "attributes_climate_ERA5Land.csv",
        "attributes_climate_obs.csv",
        "attributes_dam.csv",
        "attributes_HydroATLAS.csv",
    ]
    _TS_REL  = "CAMELS_SK/timeseries/timeseries"
    _ATT_REL = "CAMELS_SK"
    """CAMELSH_KR dataset class extending RainfallRunoff.

    This class provides access to the CAMELSH_KR dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    def __init__(
        self,
        uri: str,
        region: Optional[str] = None,
        download: bool = False,
        cache_path: Optional[str] = None,
    ) -> None:
        """Initialize CAMELSH_KR dataset.

        Args:
            uri: Path to the data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
            cache_path: Path to the cache directory
        """
        super().__init__(uri, cache_path=cache_path)
        self.region = region
        self.download = download
        if not str(uri).startswith("s3://"):
            # In aqua_fetch, CAMELS_SK is the alias of CAMELSH_KR
            self.aqua_fetch = CAMELS_SK(uri)

    def read_object_ids(self) -> np.ndarray:
        uri = str(self.data_source_dir).rstrip("/")
        if self._is_cloud():
            fs = self._make_s3fs()
            names = [p.split("/")[-1] for p in fs.ls(f"{uri}/{self._TS_REL}".removeprefix("s3://"))]
        else:
            names = os.listdir(os.path.join(uri, *self._TS_REL.split("/")))
        ids = sorted(n.replace(".csv", "") for n in names if n.endswith(".csv"))
        return np.array(ids)

    def cache_attributes_to_zarr(self) -> None:
        import zarr
        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")
        base = f"{uri}/{self._ATT_REL}"
        dfs = []
        for fname in self._ATTR_FILES:
            path = f"{base}/{fname}".removeprefix("s3://")
            try:
                with fs.open(path) as fh:
                    df = pd.read_csv(fh, index_col="STAID", dtype={"STAID": str})
                df.index = df.index.astype(str)
                # some files (e.g. attributes_HydroATLAS.csv) carry duplicate
                # STAID rows; drop them so concat keeps a unique index
                df = df[~df.index.duplicated(keep="first")]
                dfs.append(df)
            except Exception as e:
                print(f"  WARN {fname}: {e}")
        static = pd.concat(dfs, axis=1)
        static = static.loc[~static.index.duplicated(keep="first")]
        stations = self.read_object_ids().tolist()
        static = static.reindex(stations)
        static.columns = self._clean_feature_names(list(static.columns))
        static = static.rename(columns={"area": "area_km2"})
        # ERA5Land and obs climate files share column names (p_mean, frac_snow,
        # ...); AquaFetch keeps the first occurrence, so match that.
        static = static.loc[:, ~static.columns.duplicated(keep="first")]

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
        ts_base = f"{uri}/{self._TS_REL}"

        stations = self.read_object_ids().tolist()
        # hourly data: 2000-01-01 00:00 to 2019-12-31 23:00
        all_times = pd.date_range("2000-01-01", "2019-12-31 23:00", freq="h")
        n, nt = len(stations), len(all_times)
        times_ns = all_times.asi8
        all_vars = list(self._COL_MAP.values())

        zarr_name = self._timeseries_cache_filename.replace(".nc", ".zarr")
        out, opts = self._zarr_path_and_opts(zarr_name)
        root = zarr.open_group(out, mode="w", storage_options=opts, zarr_format=2)

        chunk_t = min(nt, 8760)
        time_arr = root.create_array("time", shape=(nt,), chunks=(chunk_t,), dtype="int64")
        time_arr[:] = times_ns
        time_arr.attrs["_ARRAY_DIMENSIONS"] = ["time"]
        time_arr.attrs["units"] = "nanoseconds since 1970-01-01"
        time_arr.attrs["calendar"] = "proleptic_gregorian"

        basin_arr = root.create_array("basin", shape=(n,), chunks=(n,), dtype=str)
        basin_arr[:] = stations
        basin_arr.attrs["_ARRAY_DIMENSIONS"] = ["basin"]

        for vn in all_vars:
            arr = root.create_array(vn, shape=(n, nt), chunks=(min(n, 50), chunk_t), dtype="float64")
            arr.attrs["_ARRAY_DIMENSIONS"] = ["basin", "time"]

        root.attrs["coordinates"] = "basin time"

        # fill one variable at a time
        raw_cols = list(self._COL_MAP.keys())
        zarr_vns = list(self._COL_MAP.values())
        for var_idx, (raw_col, zarr_vn) in enumerate(zip(raw_cols, zarr_vns)):
            print(f"[{var_idx+1}/{len(raw_cols)}] Reading {zarr_vn}...")
            data = np.full((n, nt), np.nan, dtype="float64")
            for i, stn in enumerate(tqdm(stations, desc=zarr_vn)):
                path = f"{ts_base}/{stn}.csv".removeprefix("s3://")
                try:
                    with fs.open(path) as fh:
                        df = pd.read_csv(fh, usecols=["DateTime", raw_col],
                                         index_col="DateTime", parse_dates=True,
                                         date_format="%d-%b-%Y %H:%M:%S")
                    df = df.reindex(all_times)
                    data[i] = pd.to_numeric(df[raw_col], errors="coerce").values
                except Exception as e:
                    print(f"  WARN {stn}: {e}")
            root[zarr_vn][:] = data
            del data

        print(f"Timeseries zarr written to: {out}")

    @property
    def _attributes_cache_filename(self):
        return "camels_sk_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camels_sk_timeseries.nc"

    @property
    def default_t_range(self):
        return ["2000-01-01", "2019-12-31"]

    # not find information of features
    _subclass_static_definitions = {
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
    }

    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "obs",
            "sources": {
                "obs": {"specific_name": "q_cms_obs", "unit": "m^3/s"},
            },
        },
        StandardVariable.WATER_LEVEL: {
            "default_source": "obs",
            "sources": {
                "obs": {"specific_name": "water_level", "unit": "m"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "total_precipitation", "unit": "mm/day"},
                "obs": {"specific_name": "precip_obs", "unit": "mm/day"},
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "temperature_2m", "unit": "°C"},
                "obs": {"specific_name": "air_temp_obs", "unit": "°C"},
                "dewpoint": {"specific_name": "dewpoint_temperature_2m", "unit": "°C"},
            },
        },
        StandardVariable.VAPOR_PRESSURE: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "dewpoint_temperature_2m", "unit": "°C"},
            },
        },
        StandardVariable.SNOW_DEPTH: {
            "default_source": "era5_depth",
            "sources": {
                "era5_depth": {"specific_name": "snow_depth", "unit": "m"},
            },
        },
        StandardVariable.SNOW_COVER: {
            "default_source": "era5_cover",
            "sources": {
                "era5_cover": {"specific_name": "snow_cover", "unit": "fraction"},
            },
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "potential_evaporation", "unit": "mm/day"},
            },
        },
        StandardVariable.U_WIND_SPEED: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "u_component_of_wind_10m", "unit": "m/s"},
            },
        },
        StandardVariable.V_WIND_SPEED: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "v_component_of_wind_10m", "unit": "m/s"},
            },
        },
        StandardVariable.WIND_SPEED: {
            "default_source": "obs_speed",
            "sources": {
                "obs_speed": {"specific_name": "wind_sp_obs", "unit": "m/s"},
            },
        },
        StandardVariable.WIND_DIR: {
            "default_source": "obs_dir",
            "sources": {
                "obs_dir": {"specific_name": "wind_dir_obs", "unit": "degree"},
            },
        },
        StandardVariable.SURFACE_PRESSURE: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "surface_pressure", "unit": "Pa"},
            },
        },
        StandardVariable.THERMAL_RADIATION: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "surface_net_thermal_radiation",
                    "unit": "W/m^2",
                },
            },
        },
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "surface_net_solar_radiation",
                    "unit": "W/m^2",
                },
            },
        },
    }
