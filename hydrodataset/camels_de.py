import os

import numpy as np
import pandas as pd
from aqua_fetch import CAMELS_DE
from hydrodataset import HydroDataset, StandardVariable
from tqdm import tqdm
from hydroutils import hydro_file


class CamelsDe(HydroDataset):
    """CAMELS-DE dataset class extending RainfallRunoff.

    This class provides access to the CAMELS-DE dataset, which contains hourly
    hydrological and meteorological data for various watersheds.
    """

    _COL_MAP = {
        "discharge_vol_obs":       "q_cms_obs",
        "discharge_spec_obs":      "q_mm_obs",
        "water_level_obs":         "water_level_obs",
        "precipitation_mean":      "pcp_mm_mean",
        "precipitation_min":       "pcp_mm_min",
        "precipitation_median":    "pcp_mm_median",
        "precipitation_max":       "pcp_mm_max",
        "precipitation_stdev":     "pcp_mm_std",
        "humidity_mean":           "rh_",
        "humidity_min":            "rh__min",
        "humidity_median":         "rh__med",
        "humidity_max":            "rh__max",
        "humidity_stdev":          "rh__std",
        "radiation_global_mean":   "solrad_wm2_mean",
        "radiation_global_min":    "solrad_wm2_min",
        "radiation_global_median": "solrad_wm2_med",
        "radiation_global_max":    "solrad_wm2_max",
        "radiation_global_stdev":  "solrad_wm2_std",
        "temperature_mean":        "airtemp_c_mean",
        "temperature_min":         "airtemp_c_min",
        "temperature_max":         "airtemp_c_max",
    }
    _ATTR_FILES = [
        "CAMELS_DE_topographic_attributes.csv",
        "CAMELS_DE_climatic_attributes.csv",
        "CAMELS_DE_humaninfluence_attributes.csv",
        "CAMELS_DE_hydrogeology_attributes.csv",
        "CAMELS_DE_hydrologic_attributes.csv",
        "CAMELS_DE_landcover_attributes.csv",
        "CAMELS_DE_soil_attributes.csv",
    ]

    def __init__(self, uri, region=None, download=False, cache_path=None):
        """Initialize CAMELS-DE dataset."""
        super().__init__(uri, cache_path=cache_path)
        self.region = region
        self.download = download
        if str(uri).startswith("s3://"):
            return
        try:
            self.aqua_fetch = CAMELS_DE(uri)
        except Exception:
            check_zip_extract = False
            zip_files = ["camels_de.zip"]
            for filename in tqdm(zip_files, desc="Checking zip files"):
                extracted_dir = self.data_source_dir.joinpath(
                    "CAMELS_DE", filename[:-4]
                )
                if not extracted_dir.exists():
                    check_zip_extract = True
                    break
            if check_zip_extract:
                hydro_file.zip_extract(self.data_source_dir.joinpath("CAMELS_DE"))
            self.aqua_fetch = CAMELS_DE(uri)

    def read_object_ids(self) -> np.ndarray:
        uri = str(self.data_source_dir).rstrip("/")
        ts_rel = "CAMELS_DE/camels_de/timeseries"
        if self._is_cloud():
            fs = self._make_s3fs()
            names = [p.split("/")[-1] for p in fs.ls(f"{uri}/{ts_rel}".removeprefix("s3://"))]
        else:
            names = os.listdir(os.path.join(uri, *ts_rel.split("/")))
        ids = sorted(
            n.replace("CAMELS_DE_hydromet_timeseries_", "").replace(".csv", "")
            for n in names if n.startswith("CAMELS_DE_hydromet_timeseries_")
        )
        return np.array(ids)

    def cache_attributes_to_zarr(self) -> None:
        import zarr
        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")
        attr_base = f"{uri}/CAMELS_DE/camels_de"
        dfs = []
        for fname in self._ATTR_FILES:
            path = f"{attr_base}/{fname}".removeprefix("s3://")
            try:
                with fs.open(path) as fh:
                    df = pd.read_csv(fh, index_col="gauge_id", dtype={"gauge_id": str})
                df.index = df.index.astype(str)
                dfs.append(df)
            except Exception as e:
                print(f"  WARN {fname}: {e}")
        static = pd.concat(dfs, axis=1)
        static = static.loc[~static.index.duplicated(keep="first")]
        static.columns = self._clean_feature_names(list(static.columns))
        static = static.rename(columns={"area": "area_km2", "gauge_lat": "lat"})

        zarr_name = self._attributes_cache_filename.replace(".nc", ".zarr")
        out, opts = self._zarr_path_and_opts(zarr_name)
        ids = static.index.tolist()
        n = len(ids)
        root = zarr.open_group(out, mode="w", storage_options=opts, zarr_format=2)
        for col in static.columns:
            vals = static[col].values.astype(str) if static[col].dtype == object else static[col].values
            arr = root.create_array(col, shape=(n,), chunks=(n,), dtype=vals.dtype)
            arr[:] = vals
            arr.attrs["_ARRAY_DIMENSIONS"] = ["basin"]
        basin_arr = root.create_array("basin", shape=(n,), chunks=(n,), dtype=str)
        basin_arr[:] = ids
        basin_arr.attrs["_ARRAY_DIMENSIONS"] = ["basin"]
        root.attrs["coordinates"] = "basin"
        self._write_zarr_units(root, "static")
        print(f"Attributes zarr written to: {out}")

    def cache_timeseries_to_zarr(self) -> None:
        import zarr
        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")
        ts_base = f"{uri}/CAMELS_DE/camels_de/timeseries"

        stations = self.read_object_ids().tolist()
        all_times = pd.date_range(self.default_t_range[0], self.default_t_range[1], freq="D")
        n, nt = len(stations), len(all_times)
        times_ns = all_times.asi8
        all_vars = list(self._COL_MAP.values())

        data: dict[str, np.ndarray] = {vn: np.full((n, nt), np.nan) for vn in all_vars}
        print(f"Reading {n} stations from OSS...")
        for i, stn in enumerate(tqdm(stations, desc="CAMELS_DE zarr")):
            path = f"{ts_base}/CAMELS_DE_hydromet_timeseries_{stn}.csv".removeprefix("s3://")
            try:
                with fs.open(path) as fh:
                    df = pd.read_csv(fh, index_col="date", parse_dates=True)
                df = df.reindex(all_times)
                for raw_col, zarr_vn in self._COL_MAP.items():
                    if raw_col in df.columns:
                        data[zarr_vn][i] = df[raw_col].values.astype(float)
            except Exception as e:
                print(f"  WARN {stn}: {e}")

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
        self._write_zarr_units(root, "dynamic")
        print(f"Timeseries zarr written to: {out}")

    @property
    def _attributes_cache_filename(self):
        return "camels_de_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camels_de_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1951-01-01", "2020-12-31"]

    # get the information of features from dataset file"CAMELS_DE_Data_Description.pdf"
    _subclass_static_definitions = {
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
        "gauge_lat": {"specific_name": "lat", "unit": "degree"},
        "gauge_lon": {"specific_name": "long", "unit": "degree"},
        "elev_mean": {"specific_name": "elev_mean", "unit": "m"},
    }

    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "vol",
            "sources": {
                "vol": {"specific_name": "q_cms_obs", "unit": "m^3/s"},
                "specific": {"specific_name": "q_mm_obs", "unit": "mm/day"},
            },
        },
        StandardVariable.WATER_LEVEL: {
            "default_source": "federal",
            "sources": {
                "federal": {"specific_name": "water_level", "unit": "m"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "pcp_mm_mean", "unit": "mm/day"},
            },
        },
        StandardVariable.PRECIPITATION_MIN: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "pcp_mm_min", "unit": "mm/day"},
            },
        },
        StandardVariable.PRECIPITATION_MAX: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "pcp_mm_max", "unit": "mm/day"},
            },
        },
        StandardVariable.PRECIPITATION_MEDIAN: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "pcp_mm_median", "unit": "mm/day"},
            },
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "airtemp_c_max", "unit": "掳C"},
            },
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "airtemp_c_min", "unit": "掳C"},
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "airtemp_c_mean", "unit": "掳C"},
            },
        },
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "solrad_wm2_mean", "unit": "W/m^2"},
            },
        },
        StandardVariable.SOLAR_RADIATION_MIN: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "solrad_wm2_min", "unit": "W/m^2"},
            },
        },
        StandardVariable.SOLAR_RADIATION_MAX: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "solrad_wm2_max", "unit": "W/m^2"},
            },
        },
        StandardVariable.SOLAR_RADIATION_MEDIAN: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "solrad_wm2_med", "unit": "W/m^2"},
            },
        },
        StandardVariable.RELATIVE_HUMIDITY: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "rh_", "unit": "%"},
            },
        },
        StandardVariable.RELATIVE_HUMIDITY_MIN: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "rh__min", "unit": "%"},
            },
        },
        StandardVariable.RELATIVE_HUMIDITY_MAX: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "rh__max", "unit": "%"},
            },
        },
        StandardVariable.RELATIVE_HUMIDITY_MEDIAN: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "rh__med", "unit": "%"},
            },
        },
    }
