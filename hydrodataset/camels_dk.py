"""
Author: Yimeng Zhang
Date: 2025-10-19 19:40:08
LastEditTime: 2025-10-19 19:54:34
LastEditors: Wenyu Ouyang
Description: CAMELS_DK dataset class
FilePath: \hydrodataset\hydrodataset\camels_dk.py
Copyright (c) 2021-2026 Wenyu Ouyang. All rights reserved.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
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

    _COL_MAP = {
        "precipitation": "pcp_mm",
        "temperature":   "airtemp_c_mean",
        "pet":           "pet_mm",
        "DKM_dtp":       "dkm_dtp",
        "DKM_eta":       "aet_mm",
        "DKM_wcr":       "dkm_wcr",
        "DKM_sdr":       "dkm_sdr",
        "DKM_sre":       "dkm_sre",
        "DKM_gwh":       "dkm_gwh",
        "Qdkm":          "qdkm",
        "DKM_irr":       "dkm_irr",
        "Abstraction":   "abstraction",
        "Qobs":          "q_cms_obs",
    }
    # AquaFetch _static_data combines only these five (signature_obs/sim are
    # excluded: they share Q_mean/Q5/Q95 column names and would collide).
    # soil's first column is "Id15_model" (values match catch_id), so every
    # file is read with index_col=0.
    _ATTR_FILES = [
        "CAMELS_DK_topography.csv",
        "CAMELS_DK_climate.csv",
        "CAMELS_DK_geology.csv",
        "CAMELS_DK_landuse.csv",
        "CAMELS_DK_soil.csv",
    ]

    def __init__(
        self, uri: str, region: Optional[str] = None, download: bool = False
    ) -> None:
        super().__init__(uri)
        self.region = region
        self.download = download
        if not str(uri).startswith("s3://"):
            self.aqua_fetch = CAMELS_DK(uri)

    def read_object_ids(self) -> np.ndarray:
        uri = str(self.data_source_dir).rstrip("/")
        ts_rel = "CAMELS_DK/Gauged_catchments/Gauged_catchments"
        if self._is_cloud():
            fs = self._make_s3fs()
            names = [p.split("/")[-1] for p in fs.ls(f"{uri}/{ts_rel}".removeprefix("s3://"))]
        else:
            names = os.listdir(os.path.join(uri, *ts_rel.split("/")))
        ids = sorted(
            n.replace("CAMELS_DK_obs_based_", "").replace(".csv", "")
            for n in names if n.startswith("CAMELS_DK_obs_based_")
        )
        return np.array(ids)

    def cache_attributes_to_zarr(self) -> None:
        import zarr
        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")
        dfs = []
        for fname in self._ATTR_FILES:
            path = f"{uri}/CAMELS_DK/{fname}".removeprefix("s3://")
            try:
                with fs.open(path) as fh:
                    df = pd.read_csv(fh, index_col=0)
                df.index = df.index.astype(str)
                dfs.append(df)
            except Exception as e:
                print(f"  WARN {fname}: {e}")
        static = pd.concat(dfs, axis=1)
        static = static.loc[~static.index.duplicated(keep="first")]
        stations = self.read_object_ids().tolist()
        static = static.reindex(stations)
        static.columns = self._clean_feature_names(list(static.columns))
        static = static.rename(columns={"catch_area": "area_km2", "catch_outlet_lat": "lat"})

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
        ts_base = f"{uri}/CAMELS_DK/Gauged_catchments/Gauged_catchments"

        stations = self.read_object_ids().tolist()
        all_times = pd.date_range(self.default_t_range[0], self.default_t_range[1], freq="D")
        n, nt = len(stations), len(all_times)
        times_ns = all_times.asi8
        all_vars = list(self._COL_MAP.values())

        data: dict[str, np.ndarray] = {vn: np.full((n, nt), np.nan) for vn in all_vars}
        print(f"Reading {n} stations from OSS...")
        for i, stn in enumerate(tqdm(stations, desc="CAMELS_DK zarr")):
            path = f"{ts_base}/CAMELS_DK_obs_based_{stn}.csv".removeprefix("s3://")
            try:
                with fs.open(path) as fh:
                    df = pd.read_csv(fh, index_col="time", parse_dates=True)
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
                "dmi": {"specific_name": "airtemp_c_mean", "unit": "掳C"},
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
