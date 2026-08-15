import os
from typing import Optional

import numpy as np
import pandas as pd
from hydrodataset import HydroDataset, StandardVariable
from aqua_fetch import CAMELS_FI


class CamelsFi(HydroDataset):
    """CAMELS_FI dataset class extending RainfallRunoff.

    This class provides access to the CAMELS_FI dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    _COL_MAP = {
        "discharge_vol":   "q_cms_obs",
        "discharge_spec":  "q_mm_obs",
        "precipitation":   "pcp_mm",
        "pet":             "pet_mm",
        "pe_era5_land":    "pe_era5_land",
        "pet_fmi":         "pet_fmi",
        "snow_evaporation":"snow_evaporation",
        "swe":             "swe_mm_era5",
        "swe_cci3-1":      "swe_mm_cci31",
        "snow_depth":      "snowdepth_m",
        "temperature_gmin":"temperature_gmin",
        "temperature_min": "airtemp_c_min",
        "temperature_mean":"airtemp_c_mean",
        "temperature_max": "airtemp_c_max",
        "humidity_rel":    "rh_",
        "radiation_global":"radiation_global",
    }
    _ATTR_FILES = [
        "CAMELS_FI_topographic_attributes.csv",
        "CAMELS_FI_climatic_attributes.csv",
        "CAMELS_FI_humaninfluence_attributes.csv",
        "CAMELS_FI_hydrologic_attributes.csv",
        "CAMELS_FI_landcover_attributes.csv",
        "CAMELS_FI_meta_attributes.csv",
        "CAMELS_FI_soil_attributes.csv",
    ]
    _DATA_REL = "CAMELS_FI/CAMELS-FI/CAMELS-FI/data"

    def __init__(
        self, uri: str, region: Optional[str] = None, download: bool = False
    ) -> None:
        super().__init__(uri)
        self.region = region
        self.download = download
        if not str(uri).startswith("s3://"):
            self.aqua_fetch = CAMELS_FI(uri)

    def read_object_ids(self) -> np.ndarray:
        uri = str(self.data_source_dir).rstrip("/")
        ts_rel = f"{self._DATA_REL}/timeseries"
        if self._is_cloud():
            fs = self._make_s3fs()
            names = [p.split("/")[-1] for p in fs.ls(f"{uri}/{ts_rel}".removeprefix("s3://"))]
        else:
            names = os.listdir(os.path.join(uri, *ts_rel.split("/")))
        ids = sorted(set(
            n.replace("CAMELS_FI_hydromet_timeseries_", "").split("_")[0]
            for n in names if n.startswith("CAMELS_FI_hydromet_timeseries_")
        ))
        return np.array(ids)

    def cache_attributes_to_zarr(self) -> None:
        import zarr
        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")
        base = f"{uri}/{self._DATA_REL}"
        dfs = []
        for fname in self._ATTR_FILES:
            path = f"{base}/{fname}".removeprefix("s3://")
            try:
                with fs.open(path) as fh:
                    df = pd.read_csv(fh, index_col="gauge_id", dtype={"gauge_id": str})
                df.index = df.index.astype(str)
                dfs.append(df)
            except Exception as e:
                print(f"  WARN {fname}: {e}")
        static = pd.concat(dfs, axis=1)
        static = static.loc[~static.index.duplicated(keep="first")]
        stations = self.read_object_ids().tolist()
        static = static.reindex(stations)
        static.columns = self._clean_feature_names(list(static.columns))
        static = static.rename(columns={"area": "area_km2", "gauge_lat": "lat"})

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
        self._write_zarr_units(root, "static")
        print(f"Attributes zarr written to: {out}")

    def cache_timeseries_to_zarr(self) -> None:
        import zarr
        from tqdm import tqdm
        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")
        ts_base = f"{uri}/{self._DATA_REL}/timeseries"

        stations = self.read_object_ids().tolist()
        all_times = pd.date_range(self.default_t_range[0], self.default_t_range[1], freq="D")
        n, nt = len(stations), len(all_times)
        times_ns = all_times.asi8
        all_vars = list(self._COL_MAP.values())

        data: dict[str, np.ndarray] = {vn: np.full((n, nt), np.nan) for vn in all_vars}
        t_start = self.default_t_range[0].replace("-", "")
        t_end = self.default_t_range[1].replace("-", "")
        print(f"Reading {n} stations from OSS...")
        for i, stn in enumerate(tqdm(stations, desc="CAMELS_FI zarr")):
            fname = f"CAMELS_FI_hydromet_timeseries_{stn}_{t_start}-{t_end}.csv"
            path = f"{ts_base}/{fname}".removeprefix("s3://")
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
        return "camels_fi_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camels_fi_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1961-01-01", "2023-12-31"]

    # get the information of features from dataset file"support_document.pdf"
    _subclass_static_definitions = {
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
        "gauge_lat": {"specific_name": "lat", "unit": "degree"},
        "gauge_lon": {"specific_name": "long", "unit": "degree"},
        "elev_mean": {"specific_name": "elev_mean", "unit": "m"},
        "pet_mean": {"specific_name": "pet_mean", "unit": "mm/day"},
    }

    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "SYKE",
            "sources": {
                "SYKE": {"specific_name": "q_cms_obs", "unit": "m^3/s"},
                "depth_based": {"specific_name": "q_mm_obs", "unit": "mm/day"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "fmi",
            "sources": {
                "fmi": {"specific_name": "pcp_mm", "unit": "mm/day"},
            },
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "default",
            "sources": {
                "default": {"specific_name": "pet_mm", "unit": "mm/day"},
                "era5_land": {"specific_name": "pe_era5_land", "unit": "mm/day"},
                "fmi": {"specific_name": "pet_fmi", "unit": "mm/day"},
            },
        },
        StandardVariable.SNOW_WATER_EQUIVALENT: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "swe_mm_era5", "unit": "mm"},
                "cci": {"specific_name": "swe_mm_cci3-1", "unit": "mm"},
            },
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "airtemp_C_min", "unit": "°C"},
                "ground_min": {"specific_name": "temperature_gmin", "unit": "°C"},
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "airtemp_C_mean", "unit": "°C"},
            },
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "airtemp_C_max", "unit": "°C"},
            },
        },
        StandardVariable.RELATIVE_HUMIDITY: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "rh_%", "unit": "%"},
            },
        },
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "radiation_global", "unit": "KJ/m^2"},
            },
        },
        StandardVariable.SNOW_DEPTH: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "snowdepth_m", "unit": "cm"},
            },
        },
    }
