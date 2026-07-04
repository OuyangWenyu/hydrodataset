import os
import xarray as xr
from typing import Union, List, Optional

from hydrodataset import HydroDataset, StandardVariable
from tqdm import tqdm
import numpy as np
import pandas as pd
from aqua_fetch import LamaHCE as _AquaFetchLamaHCE
from aqua_fetch.utils import validate_attributes


# Define custom LamaHCE class at module level to avoid pickle issues
# Named LamaHCE to maintain compatibility with file naming conventions
class LamaHCE(_AquaFetchLamaHCE):
    """
    Custom LamaHCE class that overrides fetch_static_features to have default value 'all'
    and overrides path properties to adapt to the actual dataset structure.
    """

    @property
    def data_type_dir(self):
        """Override to adapt to the actual LamaH-CE dataset structure.

        The actual structure is:
        lamaHCE/
            2_LamaH-CE_daily/
                A_basins_total_upstrm/
                B_basins_intermediate_all/
                ...
            1_LamaH-CE_daily_hourly/
                A_basins_total_upstrm/
                ...

        Original AquaFetch code expected:
        lamaHCE/
            A_basins_total_upstrm/
            B_basins_intermediate_all/
            ...
        """
        SEP = os.sep

        # Determine which parent folder based on timestep
        if self.timestep == "H":
            parent_folder = "1_LamaH-CE_daily_hourly"
        else:
            parent_folder = "2_LamaH-CE_daily"

        # Find the folder that ends with data_type
        parent_path = os.path.join(self.path, parent_folder)

        # List all directories in parent folder
        if os.path.exists(parent_path):
            dirs = [f for f in os.listdir(parent_path) if f.endswith(self.data_type)]
            if dirs:
                f = dirs[0]
                return os.path.join(parent_path, f)

        # Fallback: try original behavior if new structure doesn't exist
        dirs = [f for f in os.listdir(self.path) if f.endswith(self.data_type)]
        if dirs:
            f = dirs[0]
            return os.path.join(self.path, f)

        raise FileNotFoundError(
            f"Could not find directory ending with '{self.data_type}' "
            f"in {self.path} or {parent_path}"
        )

    @property
    def q_dir(self):
        """Override to adapt to the actual dataset structure."""
        SEP = os.sep

        # Determine which parent folder based on timestep
        if self.timestep == "H":
            parent_folder = "1_LamaH-CE_daily_hourly"
        else:
            parent_folder = "2_LamaH-CE_daily"

        # Try new structure first
        new_path = os.path.join(self.path, parent_folder, "D_gauges", "2_timeseries")
        if os.path.exists(new_path):
            return new_path

        # Fallback to original structure
        return os.path.join(self.path, "D_gauges", "2_timeseries")

    def gauge_attributes(self) -> pd.DataFrame:
        """Override to adapt to the actual dataset structure.

        Original code expected:
        lamaHCE/D_gauges/1_attributes/Gauge_attributes.csv

        Actual structure:
        lamaHCE/2_LamaH-CE_daily/D_gauges/1_attributes/Gauge_attributes.csv
        """
        # Determine which parent folder based on timestep
        if self.timestep == "H":
            parent_folder = "1_LamaH-CE_daily_hourly"
        else:
            parent_folder = "2_LamaH-CE_daily"

        # Try new structure first
        fname = os.path.join(
            self.path, parent_folder, "D_gauges", "1_attributes", "Gauge_attributes.csv"
        )

        if not os.path.exists(fname):
            # Fallback to original structure
            fname = os.path.join(
                self.path, "D_gauges", "1_attributes", "Gauge_attributes.csv"
            )

        df = pd.read_csv(fname, sep=";", index_col="ID")
        df.index = df.index.astype(str)
        return df

    def fetch_static_features(
        self,
        stations: Union[str, List[str]] = "all",
        static_features: Union[str, List[str]] = "all",  # Changed from None to 'all'
    ) -> pd.DataFrame:
        """
        static features of LamaHCE

        Modified to have default static_features='all' instead of None

        Parameters
        ----------
            stations : str
                name/id of station of which to extract the data
            static_features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Examples
        --------
            >>> from aqua_fetch import LamaHCE
            >>> dataset = LamaHCE(timestep='D', data_type='total_upstrm')
            >>> df = dataset.fetch_static_features('99')  # (1, 61)
            ...  # get list of all static features
            >>> dataset.static_features
            >>> dataset.fetch_static_features('99',
            >>> static_features=['area_calc', 'elev_mean', 'agr_fra', 'sand_fra'])  # (1, 4)
        """

        df = self.static_data()

        static_features = validate_attributes(
            static_features, self.static_features, "static features"
        )
        stations = validate_attributes(stations, self.stations(), "stations")

        df = df[static_features]

        df.index = df.index.astype(str)
        df = df.loc[stations]
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df).transpose()

        return df


class LamahCe(HydroDataset):
    """LamaHCE dataset class extending HydroDataset.

    This class provides access to the LamaHCE dataset, which contains hourly
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
        """Initialize LamaHCE dataset.

        Args:
            uri: Path to the data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
            cache_path: Path to the cache directory
        """
        super().__init__(uri, cache_path=cache_path)
        self.region = region
        self.download = download
        # cloud path: aqua_fetch cannot read S3, use cache_*_to_zarr instead
        if str(uri).startswith("s3://"):
            return
        # Use the custom LamaHCE class defined at module level
        self.aqua_fetch = LamaHCE(uri)

    # OSS relative paths (timestep=D, data_type=total_upstrm)
    _CATCH_ATTR_REL = "LamaHCE/A_basins_total_upstrm/1_attributes"
    _METEO_REL = "LamaHCE/A_basins_total_upstrm/2_timeseries/daily"
    _GAUGE_ATTR_REL = "LamaHCE/D_gauges/1_attributes"
    _Q_REL = "LamaHCE/D_gauges/2_timeseries/daily"
    # AquaFetch LamaHCE.static_map
    _STATIC_RENAME = {
        "area_calc": "area_km2",
        "slope_mean": "slope_mkm-1",
        "lon": "long",
    }
    # AquaFetch LamaHCE.dyn_map['D'] resolved to cleaned names (q file's qobs
    # is first renamed to q_cms by AquaFetch)
    _DYN_RENAME = {
        "q_cms": "q_cms_obs",
        "2m_temp_min": "airtemp_c_min",
        "2m_temp_max": "airtemp_c_max",
        "2m_temp_mean": "airtemp_c_mean",
        "prec": "pcp_mm",
        "swe": "swe_mm",
        "surf_net_solar_rad_max": "solrad_wm2_max",
        "surf_net_solar_rad_mean": "solrad_wm2",
        "surf_net_therm_rad_max": "thermrad_wm2_max",
        "surf_net_therm_rad_mean": "thermrad_wm2",
        "10m_wind_u": "windspeedu_mps",
        "10m_wind_v": "windspeedv_mps",
        "2m_dp_temp_max": "dptemp_c_max_2m",
        "2m_dp_temp_mean": "dptemp_c_mean_2m",
        "2m_dp_temp_min": "dptemp_c_min_2m",
        "surf_press": "airpres_hpa",
    }

    def read_object_ids(self) -> np.ndarray:
        if self._is_cloud():
            fs = self._make_s3fs()
            uri = str(self.data_source_dir).rstrip("/")
            names = [p.split("/")[-1] for p in fs.ls(f"{uri}/{self._METEO_REL}".removeprefix("s3://"))]
            ids = sorted(
                (n.split("_")[1].split(".csv")[0] for n in names if n.startswith("ID_")),
                key=lambda x: int(x),
            )
            return np.array(ids)
        return super().read_object_ids()

    def cache_attributes_to_zarr(self) -> None:
        import zarr

        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")

        def _read(rel, fname):
            with fs.open(f"{uri}/{rel}/{fname}".removeprefix("s3://")) as fh:
                df = pd.read_csv(fh, sep=";", index_col="ID")
            df.index = df.index.astype(str)
            return df

        cat = _read(self._CATCH_ATTR_REL, "Catchment_attributes.csv")
        gauge = _read(self._GAUGE_ATTR_REL, "Gauge_attributes.csv")
        static = pd.concat([cat, gauge], axis=1)
        static = static.loc[~static.index.duplicated(keep="first")]
        static = static.rename(columns=self._STATIC_RENAME)
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
        self._write_zarr_units(root, "static")
        print(f"Attributes zarr written to: {out}")

    def cache_timeseries_to_zarr(self) -> None:
        import zarr

        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")
        meteo_base = f"{uri}/{self._METEO_REL}"
        q_base = f"{uri}/{self._Q_REL}"

        stations = self.read_object_ids().tolist()
        all_times = pd.date_range(self.default_t_range[0], self.default_t_range[1], freq="D")
        n, nt = len(stations), len(all_times)
        times_ns = all_times.asi8

        cleaned_var_lst = []
        for info in self._dynamic_variable_mapping.values():
            for s in info["sources"].values():
                if s["specific_name"] not in cleaned_var_lst:
                    cleaned_var_lst.append(s["specific_name"])

        def _read_dated(path, q=False):
            with fs.open(path.removeprefix("s3://")) as fh:
                df = pd.read_csv(fh, sep=";")
            idx = pd.to_datetime(dict(year=df["YYYY"], month=df["MM"], day=df["DD"]))
            df = df.drop(columns=[c for c in ("YYYY", "MM", "DD", "DOY") if c in df.columns])
            df.index = idx
            if q:
                df = df.rename(columns={"qobs": "q_cms"})
            return df

        data = {vn: np.full((n, nt), np.nan) for vn in cleaned_var_lst}
        for i, stn in enumerate(tqdm(stations, desc="lamah_ce")):
            parts = []
            try:
                parts.append(_read_dated(f"{meteo_base}/ID_{stn}.csv"))
            except Exception as e:
                print(f"  WARN meteo {stn}: {e}")
            try:
                parts.append(_read_dated(f"{q_base}/ID_{stn}.csv", q=True))
            except Exception:
                pass
            if not parts:
                continue
            df = pd.concat(parts, axis=1)
            df = df.loc[~df.index.duplicated(keep="first")]
            df.columns = self._clean_feature_names(
                [self._DYN_RENAME.get(c, c) for c in df.columns]
            )
            if "airpres_hpa" in df.columns:  # AquaFetch dyn_factors: Pa -> hPa
                df["airpres_hpa"] = pd.to_numeric(df["airpres_hpa"], errors="coerce") * 0.01
            df = df.reindex(all_times)
            for vn in cleaned_var_lst:
                if vn in df.columns:
                    data[vn][i] = pd.to_numeric(df[vn], errors="coerce").values

        zarr_name = self._timeseries_cache_filename.replace(".nc", ".zarr")
        out, opts = self._zarr_path_and_opts(zarr_name)
        chunk_t = min(nt, 365)
        root = zarr.open_group(out, mode="w", storage_options=opts, zarr_format=2)
        for vn in cleaned_var_lst:
            arr = root.create_array(vn, shape=(n, nt), chunks=(min(n, 100), chunk_t),
                                    dtype="float64", fill_value=np.nan)
            arr[:] = data[vn]
            arr.attrs["_ARRAY_DIMENSIONS"] = ["basin", "time"]
        time_arr = root.create_array("time", shape=(nt,), chunks=(chunk_t,), dtype="int64")
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
        return "lamahce_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "lamahce_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1981-01-01", "2019-12-31"]

    # get the information of features from table 3 in "https://doi.org/10.5194/essd-13-4529-2021"
    # Static variable definitions based on inspected data
    _subclass_static_definitions = {
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
    }

    # Dynamic variable mapping based on inspected data
    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "q_cms_obs", "unit": "m^3/s"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "pcp_mm", "unit": "mm"},
            },
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "airtemp_c_max", "unit": "°C"},
                "dp": {"specific_name": "dptemp_c_max_2m", "unit": "°C"},
            },
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "airtemp_c_min", "unit": "°C"},
                "dp": {"specific_name": "dptemp_c_min_2m", "unit": "°C"},
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "airtemp_c_mean", "unit": "°C"},
                "dp": {"specific_name": "dptemp_c_mean_2m", "unit": "°C"},
            },
        },
        StandardVariable.EVAPOTRANSPIRATION: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "total_et", "unit": "mm"},
            },
        },
        StandardVariable.SNOW_WATER_EQUIVALENT: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "swe_mm", "unit": "mm"},
            },
        },
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "solrad_wm2", "unit": "W/m^2"},
            },
        },
        StandardVariable.SOLAR_RADIATION_MAX: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "solrad_wm2_max", "unit": "W/m^2"},
            },
        },
        StandardVariable.THERMAL_RADIATION: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "thermrad_wm2", "unit": "W/m^2"},
            },
        },
        StandardVariable.THERMAL_RADIATION_MAX: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "thermrad_wm2_max", "unit": "W/m^2"},
            },
        },
        StandardVariable.SURFACE_PRESSURE: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "airpres_hpa", "unit": "Pa"},
            },
        },
        StandardVariable.U_WIND_SPEED: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "windspeedu_mps", "unit": "m/s"},
            },
        },
        StandardVariable.V_WIND_SPEED: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "windspeedv_mps", "unit": "m/s"},
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER1: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "volsw_123", "unit": "m^3/m^3"},
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER4: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "volsw_4", "unit": "m^3/m^3"},
            },
        },
    }

    @property
    def _stations_cache_filename(self):
        """Cache filename for stations mapping."""
        return "lamahce_stations.nc"

    # Stream-network topology file (static; identical across daily/hourly bundles,
    # so the extracted 1_LamaH-CE_daily_hourly copy is used). Relative to the
    # dataset root -> works for both local (F:/data) and cloud (s3://bucket).
    _STREAM_REL = (
        "LamaHCE/1_LamaH-CE_daily_hourly/B_basins_intermediate_all/"
        "1_attributes/Stream_dist.csv"
    )

    @staticmethod
    def _prep_stream_df(df):
        """Shared processing of Stream_dist.csv (used by local NC and cloud zarr).

        Keeps NEXTDOWNID/dist_hdn/elev_diff/strm_slope, ID as string index,
        numeric columns as float. No renaming/unit conversion (matches the raw
        file), so local and cloud results are identical.
        """
        df["ID"] = df["ID"].astype(str)
        df = df.set_index("ID")[["NEXTDOWNID", "dist_hdn", "elev_diff", "strm_slope"]]
        df["NEXTDOWNID"] = df["NEXTDOWNID"].astype(str)
        for col in ["dist_hdn", "elev_diff", "strm_slope"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def cache_stations_xrdataset(self):
        """Read Stream_dist.csv (local) and cache it as NetCDF (ID-indexed).

        Columns: NEXTDOWNID (downstream station id), dist_hdn, elev_diff,
        strm_slope. Output dims/coord: ID (string).
        """
        csv_path = os.path.join(str(self.data_source_dir), *self._STREAM_REL.split("/"))
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Stream_dist.csv not found at {csv_path}")
        df = self._prep_stream_df(pd.read_csv(csv_path, sep=";"))
        output_path = self.cache_dir.joinpath(self._stations_cache_filename)
        df.to_xarray().to_netcdf(output_path)
        print(f"Stations stream data saved to: {output_path}")

    def cache_stations_to_zarr(self):
        """Read Stream_dist.csv from OSS and write the stations zarr (cloud)."""
        import zarr

        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")
        path = f"{uri}/{self._STREAM_REL}".removeprefix("s3://")
        with fs.open(path) as fh:
            df = pd.read_csv(fh, sep=";")
        df = self._prep_stream_df(df)

        zarr_name = self._stations_cache_filename.replace(".nc", ".zarr")
        out, opts = self._zarr_path_and_opts(zarr_name)
        ids = df.index.tolist()
        n = len(ids)
        root = zarr.open_group(out, mode="w", storage_options=opts, zarr_format=2)
        for col in df.columns:
            vals = df[col].values.astype(str) if df[col].dtype == object else df[col].values
            arr = root.create_array(col, shape=(n,), chunks=(n,), dtype=vals.dtype)
            arr[:] = vals
            arr.attrs["_ARRAY_DIMENSIONS"] = ["ID"]
        id_arr = root.create_array("ID", shape=(n,), chunks=(n,), dtype=str)
        id_arr[:] = ids
        id_arr.attrs["_ARRAY_DIMENSIONS"] = ["ID"]
        root.attrs["coordinates"] = "ID"
        print(f"Stations zarr written to: {out}")

    def read_stations_xrdataset(
        self,
        station_id_lst: Union[str, List[str]] = None,
    ) -> xr.Dataset:
        """Read station stream data from cached NetCDF file.

        This function reads the station stream NetCDF file and returns the
        corresponding stream attributes for the given station IDs.
        If the cache file does not exist, it will be generated first.

        Args:
            station_id_lst: A single station ID or a list of station IDs to query.
                If None, returns all stations.

        Returns:
            An xarray Dataset containing the station stream data with
            variables: NEXTDOWNID, dist_hdn, elev_diff, strm_slope.
            The dimension and coordinate is ID (station ID as string).

        Examples:
            >>> ds = lamah_ce.read_stations_xrdataset(
            ...     station_id_lst=["114", "200"]
            ... )
            >>> print(ds)
        """
        if self._is_cloud():
            import zarr as _zarr

            out, opts = self._zarr_path_and_opts(
                self._stations_cache_filename.replace(".nc", ".zarr")
            )
            try:
                ds = xr.open_zarr(out, storage_options=opts, consolidated=False,
                                  mask_and_scale=False)
            except _zarr.errors.GroupNotFoundError:
                self.cache_stations_to_zarr()
                ds = xr.open_zarr(out, storage_options=opts, consolidated=False,
                                  mask_and_scale=False)
        else:
            # Load the local cache file, generate if not exists
            cache_file = self.cache_dir.joinpath(self._stations_cache_filename)
            if not os.path.isfile(cache_file):
                self.cache_stations_xrdataset()
            ds = xr.open_dataset(cache_file)

        # Filter by station_id if provided
        if station_id_lst is not None:
            # Convert station_id_lst to list of strings
            if isinstance(station_id_lst, (str, int)):
                station_id_lst = [str(station_id_lst)]
            else:
                station_id_lst = [str(sid) for sid in station_id_lst]

            # Select stations using ID coordinate
            ds = ds.sel(ID=station_id_lst)

        return ds
