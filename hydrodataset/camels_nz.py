import os
from typing import Optional

import numpy as np
import pandas as pd
from hydrodataset import HydroDataset, StandardVariable
from tqdm import tqdm
from aqua_fetch import CAMELS_NZ as _AquaFetchCAMELS_NZ


# Define the custom CAMELS_NZ class at module level to avoid pickle errors
# Named CAMELS_NZ to maintain compatibility with file naming conventions
class CAMELS_NZ(_AquaFetchCAMELS_NZ):
    """Custom CAMELS_NZ class with updated URLs and timestep support.

    This class extends the base CAMELS_NZ to support the newer dataset version
    and provides flexible timestep configuration (hourly or daily).

    Attributes:
        timestep: Time step for the data ('H' for hourly, 'D' for daily)
    """

    # Override the base URL for the new dataset version
    url = "https://figshare.canterbury.ac.nz/ndownloader/articles/28827644/versions/2"

    def __init__(self, uri, timestep="H", **kwargs):
        """Initialize CustomCamelsNz.

        Args:
            uri: Path to the data directory
            timestep: Time step for the data ('H' for hourly, 'D' for daily)
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(uri, **kwargs)
        self.timestep = timestep

    @property
    def temp_path(self):
        folder_name = (
            "CAMELS_NZ_hourly_Temperature"
            if self.timestep == "H"
            else "CAMELS_NZ_daily_Temperature"
        )
        return os.path.join(self.path, "camels_nz", folder_name)

    @property
    def precip_path(self):
        folder_name = (
            "CAMELS_NZ_hourly_Precipitation"
            if self.timestep == "H"
            else "CAMELS_NZ_daily_Precipitation"
        )
        return os.path.join(self.path, "camels_nz", folder_name)

    @property
    def q_path(self):
        folder_name = (
            "CAMELS_NZ_hourly_Streamflow"
            if self.timestep == "H"
            else "CAMELS_NZ_daily_Streamflow"
        )
        return os.path.join(self.path, "camels_nz", folder_name)

    @property
    def shapefile_path(self):
        return os.path.join(self.path, "camels_nz", "CAMELS_NZ_Shapefiles")

    @property
    def pet_path(self):
        folder_name = (
            "CAMELS_NZ_hourly_PET" if self.timestep == "H" else "CAMELS_NZ_daily_PET"
        )
        return os.path.join(self.path, "camels_nz", folder_name)

    @property
    def rh_path(self):
        folder_name = (
            "CAMELS_NZ_hourly_Relative_Humidity"
            if self.timestep == "H"
            else "CAMELS_NZ_daily_Relative_Humidity"
        )
        return os.path.join(self.path, "camels_nz", folder_name)

    def _read_stn_dyn_para(self, stn: str, para_name: str):
        """Override _read_stn_dyn_para to handle timestep-dependent file names.

        Args:
            stn: Station ID
            para_name: Parameter name to read

        Returns:
            pandas.Series: Time series data for the station and parameter
        """
        import pandas as pd
        import numpy as np

        stn_q = pd.Series(dtype=np.float32, name=stn)

        fname = {"Relative_humidity": "RH"}

        # Construct file name based on timestep
        prefix = fname.get(para_name, para_name)
        if self.timestep == "D":
            prefix = f"daily_{prefix}"

        fpath = os.path.join(
            self._path_map[para_name], f"{prefix}_station_id_{stn}.csv"
        )

        if os.path.exists(fpath):
            if para_name == "flow" and stn in self._nodata_stns:
                return stn_q

            try:
                stn_q = pd.read_csv(
                    fpath, index_col=0, parse_dates=True, na_values=["NA  "]
                )
            except pd.errors.EmptyDataError:
                warning_prefix = "daily_" if self.timestep == "D" else ""
                print(
                    f"Warning: {warning_prefix}{para_name}_station_id_{stn}.csv is empty. Skipping station {stn}."
                )
                return stn_q

            format = "%m/%d/%Y %H:%M" if self.timestep == "H" else "%m/%d/%Y"
            if para_name == "flow" and stn == "57521" and self.timestep == "H":
                format = "%d/%m/%Y %H:%M"
            elif para_name == "flow" and stn == "57521" and self.timestep == "D":
                format = "%d/%m/%Y"

            stn_q.index = pd.to_datetime(stn_q.index, format=format)
            stn_q = stn_q[para_name].astype(np.float32).rename(stn)
        else:
            if self.verbosity > 1:
                warning_prefix = "daily_" if self.timestep == "D" else ""
                print(
                    f"Warning: {warning_prefix}{para_name}_station_id_{stn}.csv does not exist. Skipping station {stn}."
                )
            stn_q = pd.Series(dtype=np.float32, name=stn)

        # Remove rows with duplicated index
        stn_q = stn_q[~stn_q.index.duplicated(keep="first")]

        return stn_q

    def _maybe_to_netcdf(self, *args, **kwargs):
        """Override to disable netcdf conversion."""
        pass


class CamelsNz(HydroDataset):

    # (subfolder, filename_prefix, data_column, zarr_var_name)
    _VAR_MAP = [
        ("CAMELS_NZ_Streamflow",       "flow_station_id_",          "flow",              "q_cms_obs"),
        ("CAMELS_NZ_Precipitation",    "precipitation_station_id_", "precipitation",     "pcp_mm"),
        ("CAMELS_NZ_Temperature",      "temperature_station_id_",   "temperature",       "airtemp_c_mean"),
        ("CAMELS_NZ_PET",              "PET_station_id_",           "PET",               "pet_mm"),
        ("CAMELS_NZ_Relative_Humidity","RH_station_id_",            "Relative_humidity", "rh_"),
    ]
    _ATTR_FILES = [
        "1.CAMELS_NZ_Catchment_information.csv",
        "2.CAMELS_NZ_Climatic_attribute.csv",
        "3.CAMELS_NZ_Landcover_attribute.csv",
        "4.CAMELS_NZ_Geology.csv",
        "5.CAMELS_NZ_Anthropogenic_attribute.csv",
    ]
    _DATA_REL = "CAMELS_NZ/camels_nz"
    """CAMELS_NZ dataset class.

    This class uses a custom data reading implementation to support a newer
    dataset version than the one supported by the underlying aquafetch library.
    It overrides the download URLs and provides its own parsing and caching logic.

    The dataset supports both hourly ('H') and daily ('D') timesteps.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        timestep: Time step for the data ('H' for hourly, 'D' for daily)
    """

    def __init__(
        self,
        uri: str,
        region: Optional[str] = None,
        download: bool = False,
        timestep: str = "H",
    ) -> None:
        """Initialize CAMELS_NZ dataset.

        Args:
            uri: Path to the data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
            timestep: Time step for the data ('H' for hourly, 'D' for daily, default: 'H')
        """
        super().__init__(uri)
        self.region = "NZ" if region is None else region
        self.download = download
        self.timestep = timestep
        if not str(uri).startswith("s3://"):
            self.aqua_fetch = CAMELS_NZ(uri, timestep=timestep)

    def read_object_ids(self) -> np.ndarray:
        uri = str(self.data_source_dir).rstrip("/")
        flow_rel = f"{self._DATA_REL}/CAMELS_NZ_Streamflow"
        if self._is_cloud():
            fs = self._make_s3fs()
            names = [p.split("/")[-1] for p in fs.ls(f"{uri}/{flow_rel}".removeprefix("s3://"))]
        else:
            names = os.listdir(os.path.join(uri, *flow_rel.split("/")))
        ids = sorted(
            n.replace("flow_station_id_", "").replace(".csv", "")
            for n in names if n.startswith("flow_station_id_")
        )
        return np.array(ids)

    def cache_attributes_to_zarr(self) -> None:
        import zarr
        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")
        attr_base = f"{uri}/{self._DATA_REL}/CAMELS_NZ_Catchment_Atrributes"
        dfs = []
        for i, fname in enumerate(self._ATTR_FILES):
            path = f"{attr_base}/{fname}".removeprefix("s3://")
            try:
                with fs.open(path) as fh:
                    raw = fh.read()
                df = pd.read_csv(
                    __import__("io").BytesIO(raw),
                    index_col=0, dtype={0: str}, encoding="utf-8-sig",
                )
                df.index = df.index.astype(str)
                # Every file repeats RID/StationName/latitude/longitude; AquaFetch
                # keeps them only from the first file and drops them elsewhere.
                if i > 0:
                    df = df.drop(
                        columns=["RID", "StationName", "latitude", "longitude"],
                        errors="ignore",
                    )
                dfs.append(df)
            except Exception as e:
                print(f"  WARN {fname}: {e}")
        static = pd.concat(dfs, axis=1)
        static = static.loc[~static.index.duplicated(keep="first")]
        stations = self.read_object_ids().tolist()
        static = static.reindex(stations)
        static.columns = self._clean_feature_names(list(static.columns))
        static = static.rename(columns={"uparea": "area_km2"})
        # NZ has no mean-precip attribute; derive p_mean from the timeseries
        static["p_mean"] = self._p_mean_from_precip(static.index)

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
        freq = "h" if self.timestep == "H" else "D"

        stations = self.read_object_ids().tolist()
        all_times = pd.date_range(self.default_t_range[0], self.default_t_range[1], freq=freq)
        n, nt = len(stations), len(all_times)
        times_ns = all_times.asi8
        all_vars = [row[3] for row in self._VAR_MAP]

        zarr_name = self._timeseries_cache_filename.replace(".nc", ".zarr")
        out, opts = self._zarr_path_and_opts(zarr_name)
        root = zarr.open_group(out, mode="w", storage_options=opts, zarr_format=2)

        # Pre-create coordinate arrays
        time_arr = root.create_array("time", shape=(nt,), chunks=(min(nt, 8760),), dtype="int64")
        time_arr[:] = times_ns
        time_arr.attrs["_ARRAY_DIMENSIONS"] = ["time"]
        time_arr.attrs["units"] = "nanoseconds since 1970-01-01"
        time_arr.attrs["calendar"] = "proleptic_gregorian"

        basin_arr = root.create_array("basin", shape=(n,), chunks=(n,), dtype=str)
        basin_arr[:] = stations
        basin_arr.attrs["_ARRAY_DIMENSIONS"] = ["basin"]

        # Pre-create all data arrays
        chunk_t = min(nt, 8760)
        for vn in all_vars:
            arr = root.create_array(vn, shape=(n, nt), chunks=(min(n, 50), chunk_t), dtype="float64")
            arr.attrs["_ARRAY_DIMENSIONS"] = ["basin", "time"]

        root.attrs["coordinates"] = "basin time"

        # Fill one variable at a time to limit memory usage
        for subfolder, prefix, col, zarr_vn in self._VAR_MAP:
            print(f"Reading {zarr_vn} ({n} stations)...")
            data = np.full((n, nt), np.nan, dtype="float64")
            ts_base = f"{base}/{subfolder}"
            for i, stn in enumerate(tqdm(stations, desc=zarr_vn)):
                path = f"{ts_base}/{prefix}{stn}.csv".removeprefix("s3://")
                try:
                    with fs.open(path) as fh:
                        df = pd.read_csv(fh, index_col="time", parse_dates=True)
                    if col in df.columns:
                        # some station files carry duplicate timestamps, which
                        # breaks reindex; keep the first occurrence
                        df = df[~df.index.duplicated(keep="first")]
                        df = df[[col]].reindex(all_times)
                        data[i] = pd.to_numeric(df[col], errors="coerce").values
                except Exception as e:
                    print(f"  WARN {stn}: {e}")
            root[zarr_vn][:] = data
            del data
            print(f"  -> written")

        print(f"Timeseries zarr written to: {out}")

    @property
    def _attributes_cache_filename(self):
        return f"camels_nz_{self.timestep.lower()}_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return f"camels_nz_{self.timestep.lower()}_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1972-01-01", "2024-08-02"]

    # Static variable definitions for CAMELS-NZ
    # Note: specific_name should be the cleaned version (lowercase, no spaces)
    # as stored in the cache file after _clean_feature_names() processing
    _subclass_static_definitions = {
        "area": {"specific_name": "area_km2", "unit": "km^2"},
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
    }

    # Dynamic variable mapping for CAMELS-NZ
    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "obs",
            "sources": {"obs": {"specific_name": "q_cms_obs", "unit": "m^3/s"}},
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "default",
            "sources": {"default": {"specific_name": "pcp_mm", "unit": "mm/day"}},
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "default",
            "sources": {"default": {"specific_name": "airtemp_c_mean", "unit": "°C"}},
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "default",
            "sources": {"default": {"specific_name": "pet_mm", "unit": "mm/day"}},
        },
        StandardVariable.RELATIVE_HUMIDITY: {
            "default_source": "default",
            "sources": {"default": {"specific_name": "rh_", "unit": "%"}},
        },
    }
