import os
from typing import Optional

import numpy as np
import pandas as pd

from aqua_fetch import Simbi
from hydrodataset import HydroDataset, StandardVariable


class simbi(HydroDataset):
    """simbi dataset class extending RainfallRunoff.

    This class provides access to the simbi dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    # OSS dataset folder (uploaded raw data)
    _DATA_REL = "Simbi"
    _ATTR_REL = "Simbi/03_SIMBI_ATTRIBUTE"
    _OTHERS_REL = "Simbi/03_SIMBI_ATTRIBUTE/02_OTHERS"
    _DAILY_REL = "Simbi/03_SIMBI_ATTRIBUTE/01_CLIMATIC_SIGNATURE/02_DAILY"
    _MONTHLY_REL = "Simbi/03_SIMBI_ATTRIBUTE/01_CLIMATIC_SIGNATURE/01_MONTHLY"
    _Q_REL = "Simbi/00_SIMBI_OBSERVED_DATA/02_DAILY_STREAMFLOW"
    _PCP_REL = "Simbi/00_SIMBI_OBSERVED_DATA/01_DAILY_RAINFALL"
    _TEMP_REL = "Simbi/00_SIMBI_OBSERVED_DATA/05_DAILY_LONG_TERM_AVERAGE_TEMPERATURE"

    # static attribute files mirroring AquaFetch Simbi._static_data:
    # other_attributes() then clim_sigs() (daily then monthly).
    # each is (relative_dir, filename, column_suffix); every file is read with
    # index_col=0 and the index normalised via id.split("-")[1].
    _STATIC_FILES = [
        (_OTHERS_REL, "stream_density.csv", ""),
        (_OTHERS_REL, "Percent_land_cover_98.csv", "_lc_98"),
        (_OTHERS_REL, "Percent_land_cover_95.csv", "_lc_95"),
        (_OTHERS_REL, "Percent_geologic_class.csv", "_geol"),
        (_OTHERS_REL, "location_and_topography.csv", ""),
        (_OTHERS_REL, "hypsometric_curve.csv", ""),
        (_OTHERS_REL, "Percent_aquifer_class.csv", ""),
        (_OTHERS_REL, "Percent_carb_sediment_magma.csv", ""),
        (_DAILY_REL, "baseflow_index.csv", "_d"),
        (_DAILY_REL, "high_q_dur.csv", "_d_hq_dur"),
        (_DAILY_REL, "high_q_freq.csv", "_d_hq_freq"),
        (_DAILY_REL, "low_q_dur.csv", "_d_lq_dur"),
        (_DAILY_REL, "low_q_freq.csv", "_d_lq_freq"),
        (_DAILY_REL, "q_mean.csv", "_d_mean"),
        (_DAILY_REL, "quantile_5.csv", "_d_q5"),
        (_DAILY_REL, "quantile_95.csv", "_d_q95"),
        (_MONTHLY_REL, "aridity_runoff.csv", "_mon_arid"),
        (_MONTHLY_REL, "average.csv", "_mon_avg"),
        (_MONTHLY_REL, "QMNA5.csv", "_mon_QMNA5"),
        (_MONTHLY_REL, "QMXA10.csv", "_mon_QMXA10"),
        (_MONTHLY_REL, "quantile_5.csv", "_mon_q5"),
        (_MONTHLY_REL, "quantile_95.csv", "_mon_q95"),
    ]
    # AquaFetch Simbi.static_map (raw column -> standard name)
    _STATIC_RENAME = {
        "Area": "area_km2",
        "Lat_Cent": "lat",
        "Lon_Cent": "long",
        "Slope": "slope_degrees",
    }

    def __init__(
        self, uri: str, region: Optional[str] = None, download: bool = False
    ) -> None:
        """Initialize simbi dataset.

        Args:
            uri: Path to the data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
        """
        super().__init__(uri)
        self.region = region
        self.download = download
        # aqua_fetch only supports local paths
        if not str(uri).startswith("s3://"):
            self.aqua_fetch = Simbi(uri)

    def read_object_ids(self) -> np.ndarray:
        """Station IDs: the 24 catchments with boundary + static data.

        Derived from the topography attribute file index (id.split('-')[1]),
        which matches AquaFetch Simbi.boundary_stations().
        """
        if self._is_cloud():
            fs = self._make_s3fs()
            uri = str(self.data_source_dir).rstrip("/")
            path = f"{uri}/{self._OTHERS_REL}/location_and_topography.csv".removeprefix(
                "s3://"
            )
            with fs.open(path) as fh:
                idx = pd.read_csv(fh, index_col=0).index
            ids = sorted(str(i).split("-")[1] for i in idx)
            return np.array(ids)
        return super().read_object_ids()

    def cache_attributes_to_zarr(self) -> None:
        import zarr

        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")
        dfs = []
        for rel, fname, suffix in self._STATIC_FILES:
            path = f"{uri}/{rel}/{fname}".removeprefix("s3://")
            try:
                with fs.open(path) as fh:
                    df = pd.read_csv(fh, index_col=0)
                df.index = [str(i).split("-")[1] for i in df.index]
                if suffix:
                    df.columns = [f"{c}{suffix}" for c in df.columns]
                dfs.append(df)
            except Exception as e:
                print(f"  WARN {fname}: {e}")
        static = pd.concat(dfs, axis=1)
        static = static.loc[~static.index.duplicated(keep="first")]
        static = static.rename(columns=self._STATIC_RENAME)
        static.index = static.index.astype(str)
        static.columns = self._clean_feature_names(list(static.columns))
        static = static.loc[:, ~static.columns.duplicated(keep="first")]

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
        print(f"Attributes zarr written to: {out}")

    def cache_timeseries_to_zarr(self) -> None:
        import zarr
        from tqdm import tqdm

        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")
        q_base = f"{uri}/{self._Q_REL}"
        pcp_dirs = [f"{uri}/{self._PCP_REL}/1920_1940", f"{uri}/{self._PCP_REL}/1948_1966"]
        temp_base = f"{uri}/{self._TEMP_REL}"

        stations = self.read_object_ids().tolist()
        all_times = pd.date_range(self.default_t_range[0], self.default_t_range[1], freq="D")
        n, nt = len(stations), len(all_times)
        times_ns = all_times.asi8

        def _read_one(path, col_out):
            try:
                with fs.open(path.removeprefix("s3://")) as fh:
                    df = pd.read_csv(fh, index_col=0, parse_dates=True)
                s = df.iloc[:, 0]
                s.index = pd.to_datetime(s.index)
                s = s[~s.index.duplicated(keep="first")]
                return s
            except Exception:
                return None

        # cleaned zarr var names from dynamic mapping
        q_vn, pcp_vn, temp_vn = "q_cms_obs", "pcp_mm", "airtemp_c_mean"
        data = {vn: np.full((n, nt), np.nan) for vn in (q_vn, pcp_vn, temp_vn)}

        for i, stn in enumerate(tqdm(stations, desc="simbi")):
            q = _read_one(f"{q_base}/Q_{stn}.csv", q_vn)
            if q is not None:
                data[q_vn][i] = pd.to_numeric(q.reindex(all_times), errors="coerce").values
            # precipitation: concat the two period folders
            parts = [_read_one(f"{d}/P_{stn}.csv", pcp_vn) for d in pcp_dirs]
            parts = [p for p in parts if p is not None]
            if parts:
                pcp = pd.concat(parts)
                pcp = pcp[~pcp.index.duplicated(keep="first")]
                data[pcp_vn][i] = pd.to_numeric(pcp.reindex(all_times), errors="coerce").values
            t = _read_one(f"{temp_base}/P_{stn}.csv", temp_vn)
            if t is not None:
                data[temp_vn][i] = pd.to_numeric(t.reindex(all_times), errors="coerce").values

        zarr_name = self._timeseries_cache_filename.replace(".nc", ".zarr")
        out, opts = self._zarr_path_and_opts(zarr_name)
        root = zarr.open_group(out, mode="w", storage_options=opts, zarr_format=2)
        for vn in (q_vn, pcp_vn, temp_vn):
            arr = root.create_array(vn, shape=(n, nt), chunks=(n, min(nt, 365)),
                                    dtype="float64", fill_value=np.nan)
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
        return "simbi_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "simbi_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1920-01-01", "2005-12-31"]

    # get the information of features from dataset file "SIMBI_README"
    _subclass_static_definitions = {
        "area": {"specific_name": "area_km2", "unit": "km^2"},
        "p_mean": {"specific_name": "p_mon_avg", "unit": "mm/month"},
    }

    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "q_cms_obs", "unit": "m^3/s"}
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "observations",
            "sources": {"observations": {"specific_name": "pcp_mm", "unit": "mm/day"}},
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "airtemp_c_mean", "unit": "°C"}
            },
        },
    }
