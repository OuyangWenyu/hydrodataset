import os
from typing import Optional

import numpy as np
import pandas as pd
from hydrodataset import HydroDataset, StandardVariable
from aqua_fetch import CAMELS_SE


class CamelsSe(HydroDataset):

    _COL_MAP = {
        "Qobs_m3s": "q_cms_obs",
        "Qobs_mm":  "q_mm_obs",
        "Pobs_mm":  "pcp_mm",
        "Tobs_C":   "airtemp_c_mean",
    }
    # (filename, column suffix) — AquaFetch suffixes soil/signature columns so
    # that names shared across files (e.g. water_percentage, the Sxx signatures)
    # do not collide when concatenated.
    _ATTR_FILES = [
        ("catchments_physical_properties.csv", ""),
        ("catchments_landcover.csv", ""),
        ("catchments_soil_classes.csv", "_sc"),
        ("catchments_hydrological_signatures_1961_2020.csv", "_hs"),
        ("catchments_hydrological_signatures_CNP1_1961_1990.csv", "_CNP_61_90"),
        ("catchments_hydrological_signatures_CNP2_1991_2020.csv", "_CNP_91_20"),
    ]
    _TS_REL  = "CAMELS_SE/catchment time series/catchment time series"
    _ATT_REL = "CAMELS_SE/catchment properties/catchment properties"
    """CAMELS_SE dataset class extending RainfallRunoff.

    This class provides access to the CAMELS_SE dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    def __init__(
        self, uri: str, region: Optional[str] = None, download: bool = False
    ) -> None:
        """Initialize CAMELS_SE dataset.

        Args:
            uri: Path to the data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
        """
        super().__init__(uri)
        self.region = region
        self.download = download
        if not str(uri).startswith("s3://"):
            self.aqua_fetch = CAMELS_SE(uri)

    def read_object_ids(self) -> np.ndarray:
        uri = str(self.data_source_dir).rstrip("/")
        if self._is_cloud():
            fs = self._make_s3fs()
            names = [p.split("/")[-1] for p in fs.ls(f"{uri}/{self._TS_REL}".removeprefix("s3://"))]
        else:
            names = os.listdir(os.path.join(uri, *self._TS_REL.split("/")))
        ids = sorted(
            (n.replace("catchment_id_", "").split("_")[0]
             for n in names if n.startswith("catchment_id_")),
            key=lambda x: int(x),
        )
        return np.array(ids)

    def cache_attributes_to_zarr(self) -> None:
        import zarr
        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")
        base = f"{uri}/{self._ATT_REL}"
        dfs = []
        for fname, suffix in self._ATTR_FILES:
            path = f"{base}/{fname}".removeprefix("s3://")
            try:
                with fs.open(path) as fh:
                    df = pd.read_csv(fh, index_col="ID", dtype={"ID": str})
                df.index = df.index.astype(str)
                if suffix:
                    df.columns = [f"{c}{suffix}" for c in df.columns]
                dfs.append(df)
            except Exception as e:
                print(f"  WARN {fname}: {e}")
        static = pd.concat(dfs, axis=1)
        static = static.loc[~static.index.duplicated(keep="first")]
        stations = self.read_object_ids().tolist()
        static = static.reindex(stations)
        static.columns = self._clean_feature_names(list(static.columns))
        static = static.rename(columns={"latitude_wgs84": "lat"})

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
        ts_base = f"{uri}/{self._TS_REL}"

        # Build a map from station ID 鈫?filename (IDs can have spaces in name part)
        if self._is_cloud():
            fs2 = self._make_s3fs()
            all_names = [p.split("/")[-1] for p in fs2.ls(ts_base.removeprefix("s3://"))]
        else:
            all_names = os.listdir(os.path.join(uri, *self._TS_REL.split("/")))
        id_to_fname = {
            n.replace("catchment_id_", "").split("_")[0]: n
            for n in all_names if n.startswith("catchment_id_")
        }

        stations = self.read_object_ids().tolist()
        all_times = pd.date_range(self.default_t_range[0], self.default_t_range[1], freq="D")
        n, nt = len(stations), len(all_times)
        times_ns = all_times.asi8
        all_vars = list(self._COL_MAP.values())

        data: dict[str, np.ndarray] = {vn: np.full((n, nt), np.nan) for vn in all_vars}
        print(f"Reading {n} stations from OSS...")
        for i, stn in enumerate(tqdm(stations, desc="CAMELS_SE zarr")):
            fname = id_to_fname.get(stn)
            if fname is None:
                continue
            path = f"{ts_base}/{fname}".removeprefix("s3://")
            try:
                with fs.open(path) as fh:
                    df = pd.read_csv(fh)
                df.index = pd.to_datetime(
                    {"year": df["Year"], "month": df["Month"], "day": df["Day"]}
                )
                df = df.reindex(all_times)
                for raw_col, zarr_vn in self._COL_MAP.items():
                    if raw_col in df.columns:
                        data[zarr_vn][i] = pd.to_numeric(df[raw_col], errors="coerce").values
            except Exception as e:
                print(f"  WARN {stn}: {e}")

        zarr_name = self._timeseries_cache_filename.replace(".nc", ".zarr")
        out, opts = self._zarr_path_and_opts(zarr_name)
        root = zarr.open_group(out, mode="w", storage_options=opts, zarr_format=2)
        for vn in all_vars:
            arr = root.create_array(vn, shape=(n, nt), chunks=(n, min(nt, 365)), dtype="float64")
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
        return "camels_se_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camels_se_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1961-01-01", "2020-12-31"]

    # get the information of features from dataset file"Documentation_2024-01-02.pdf"
    _subclass_static_definitions = {
        "p_mean": {"specific_name": "pmean_mm_year", "unit": "mm/year"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
        "urban_percentage": {"specific_name": "urban_percentage", "unit": "%"},
    }

    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "obs_cms",
            "sources": {
                "obs_cms": {"specific_name": "q_cms_obs", "unit": "m^3/s"},
                "obs_mm": {"specific_name": "q_mm_obs", "unit": "mm/day"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "default",
            "sources": {
                "default": {"specific_name": "pcp_mm", "unit": "mm/day"},
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "default",
            "sources": {
                "default": {"specific_name": "airtemp_C_mean", "unit": "掳C"},
            },
        },
    }
