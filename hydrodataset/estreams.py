import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
from typing import Optional

from tqdm import tqdm
from aqua_fetch import EStreams
from hydrodataset import HydroDataset, StandardVariable


class Estreams(HydroDataset):
    """EStreams dataset class extending HydroDataset.

    This class uses a custom data reading implementation to support a newer
    dataset version than the one supported by the underlying aquafetch library.
    It overrides the download URLs and provides updated methods.
    """

    def __init__(
        self, uri: str, region: Optional[str] = None, download: bool = False
    ) -> None:
        """Initialize EStreams dataset.

        Args:
            uri: Path to the data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
        """
        super().__init__(uri)
        self.region = region
        self.download = download

        # cloud path: aqua_fetch cannot read S3, use cache_*_to_zarr instead
        if str(uri).startswith("s3://"):
            return
        # Instantiate EStreams from aqua_fetch
        # The _read_stn_dyn method and path2 fix have been added directly to aqua_fetch
        self.aqua_fetch = EStreams(uri)

    # OSS relative paths (folder EStreams; aqua path2 = path/EStreams/EStreams)
    _P2 = "EStreams/EStreams/EStreams"
    _STATIC_REL = "EStreams/EStreams/EStreams/attributes/static_attributes"
    _SIG_REL = "EStreams/EStreams/EStreams/hydroclimatic_signatures"
    _GAUGE_REL = "EStreams/EStreams/EStreams/streamflow_gauges"
    _METEO_REL = "EStreams/EStreams/EStreams/meteorology"
    # AquaFetch EStreams.static_map
    _STATIC_RENAME = {
        "area_estreams": "area_km2",
        "slope_sawicz": "slope_no_unit",
        "lon": "long",
    }
    # AquaFetch EStreams.dyn_map resolved to cleaned names (others pass through,
    # e.g. sp_mean stays sp_mean)
    _DYN_RENAME = {
        "t_min": "airtemp_c_min",
        "t_max": "airtemp_c_max",
        "t_mean": "airtemp_c_mean",
        "p_mean": "pcp_mm",
        "pet_mean": "pet_mm",
        "rh_mean": "rh_",
        "swr_mean": "solrad_wm2",
        "ws_mean": "windspeed_mps",
    }

    def read_object_ids(self) -> np.ndarray:
        if self._is_cloud():
            fs = self._make_s3fs()
            uri = str(self.data_source_dir).rstrip("/")
            names = [p.split("/")[-1] for p in fs.ls(f"{uri}/{self._METEO_REL}".removeprefix("s3://"))]
            ids = sorted(
                n[len("estreams_meteorology_"):].split(".csv")[0]
                for n in names if n.startswith("estreams_meteorology_")
            )
            return np.array(ids)
        return super().read_object_ids()

    def cache_attributes_to_zarr(self) -> None:
        import zarr

        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")

        def _read(rel, fname):
            with fs.open(f"{uri}/{rel}/{fname}".removeprefix("s3://")) as fh:
                df = pd.read_csv(fh, index_col="basin_id", dtype={"basin_id": str})
            df.index = df.index.astype(str)
            return df

        dfs = [
            _read(self._SIG_REL, "estreams_hydrometeo_signatures.csv"),
            _read(self._GAUGE_REL, "estreams_gauging_stations.csv"),
        ]
        sdir = f"{uri}/{self._STATIC_REL}".removeprefix("s3://")
        for p in sorted(fs.ls(sdir)):
            if p.endswith(".csv"):
                with fs.open(p) as fh:
                    df = pd.read_csv(fh, index_col="basin_id", dtype={"basin_id": str})
                df.index = df.index.astype(str)
                dfs.append(df)
        static = pd.concat(dfs, axis=1)
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
        print(f"Attributes zarr written to: {out}")

    def cache_timeseries_to_zarr(self, batch_size: int = 500) -> None:
        import zarr

        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")
        ts_base = f"{uri}/{self._METEO_REL}"

        stations = self.read_object_ids().tolist()
        n = len(stations)
        all_times = pd.date_range(self.default_t_range[0], self.default_t_range[1], freq="D")
        nt = len(all_times)
        times_ns = all_times.asi8

        cleaned_var_lst = []
        for info in self._dynamic_variable_mapping.values():
            for s in info["sources"].values():
                if s["specific_name"] not in cleaned_var_lst:
                    cleaned_var_lst.append(s["specific_name"])

        zarr_name = self._timeseries_cache_filename.replace(".nc", ".zarr")
        out, opts = self._zarr_path_and_opts(zarr_name)
        chunk_t = min(nt, 365)
        chunk_b = min(batch_size, n)
        root = zarr.open_group(out, mode="a", storage_options=opts, zarr_format=2)
        if "basin" not in root:
            for vn in cleaned_var_lst:
                arr = root.create_array(vn, shape=(n, nt), chunks=(chunk_b, chunk_t),
                                        dtype="float64", fill_value=np.nan)
                arr.attrs["_ARRAY_DIMENSIONS"] = ["basin", "time"]
            time_arr = root.create_array("time", shape=(nt,), chunks=(chunk_t,), dtype="int64")
            time_arr[:] = times_ns
            time_arr.attrs["_ARRAY_DIMENSIONS"] = ["time"]
            time_arr.attrs["units"] = "nanoseconds since 1970-01-01"
            time_arr.attrs["calendar"] = "proleptic_gregorian"
            basin_arr = root.create_array("basin", shape=(n,), chunks=(n,), dtype=str)
            basin_arr[:] = stations
            basin_arr.attrs["_ARRAY_DIMENSIONS"] = ["basin"]
            prog = root.create_array("_progress", shape=(n,), chunks=(n,), dtype="int8", fill_value=0)
            prog[:] = 0
            prog.attrs["_ARRAY_DIMENSIONS"] = ["basin"]
            root.attrs["coordinates"] = "basin time"
        progress = root["_progress"]

        n_batches = (n + batch_size - 1) // batch_size
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            bnum = start // batch_size + 1
            if all(progress[start:end]):
                print(f"Batch {bnum}/{n_batches}: already done, skipping")
                continue
            print(f"Batch {bnum}/{n_batches}: {end-start} stations ...")
            buffers = {vn: np.full((end - start, nt), np.nan) for vn in cleaned_var_lst}
            for j, stn in enumerate(tqdm(stations[start:end], desc=f"batch {bnum}")):
                path = f"{ts_base}/estreams_meteorology_{stn}.csv".removeprefix("s3://")
                try:
                    with fs.open(path) as fh:
                        df = pd.read_csv(fh, index_col="date", parse_dates=True)
                    df = df.rename(columns=self._DYN_RENAME)
                    df = df[~df.index.duplicated(keep="first")]
                    df.columns = self._clean_feature_names(list(df.columns))
                    df = df.reindex(all_times)
                    for vn in cleaned_var_lst:
                        if vn in df.columns:
                            buffers[vn][j] = pd.to_numeric(df[vn], errors="coerce").values
                except Exception as e:
                    print(f"  WARN {stn}: {e}")
            for vn in cleaned_var_lst:
                root[vn][start:end, :] = buffers[vn]
            progress[start:end] = 1
            print(f"Batch {bnum}/{n_batches}: done")
        print(f"Timeseries zarr written to: {out}")

    @property
    def _attributes_cache_filename(self):
        return "estreams_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "estreams_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1950-01-01", "2023-06-30"]

    # get the information of features from "https://www.nature.com/articles/s41597-024-03706-1/tables/6"
    # Define standardized static variable mappings
    _subclass_static_definitions = {
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
    }

    # Define standardized dynamic variable mappings
    _dynamic_variable_mapping = {
        StandardVariable.PRECIPITATION: {
            "default_source": "estreams",
            "sources": {"estreams": {"specific_name": "pcp_mm", "unit": "mm/day"}},
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "estreams",
            "sources": {"estreams": {"specific_name": "airtemp_c_mean", "unit": "°C"}},
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "estreams",
            "sources": {"estreams": {"specific_name": "airtemp_c_min", "unit": "°C"}},
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "estreams",
            "sources": {"estreams": {"specific_name": "airtemp_c_max", "unit": "°C"}},
        },
        StandardVariable.SURFACE_PRESSURE: {
            "default_source": "estreams",
            "sources": {"estreams": {"specific_name": "sp_mean", "unit": "hPa"}},
        },
        StandardVariable.RELATIVE_HUMIDITY: {
            "default_source": "estreams",
            "sources": {"estreams": {"specific_name": "rh_", "unit": "%"}},
        },
        StandardVariable.WIND_SPEED: {
            "default_source": "estreams",
            "sources": {"estreams": {"specific_name": "windspeed_mps", "unit": "m/s"}},
        },
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "estreams",
            "sources": {"estreams": {"specific_name": "solrad_wm2", "unit": "W/m^2"}},
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "estreams",
            "sources": {"estreams": {"specific_name": "pet_mm", "unit": "mm/day"}},
        },
    }

    def cache_timeseries_xrdataset(self, batch_size=500):
        """Cache timeseries from per-station meteorology CSV files in batches.

        The aqua_fetch ``EStreams`` class reads daily data from individual
        ``estreams_meteorology_{station}.csv`` files (9 dynamic variables,
        one file per station).  We call ``meteo_data_station()`` to get the
        correctly-renamed columns and then build variable-first xarray batches.
        """
        if not hasattr(self, "aqua_fetch"):
            raise NotImplementedError("aqua_fetch attribute is required")

        unit_lookup = {}
        if hasattr(self, "_dynamic_variable_mapping"):
            for std_name, mapping_info in self._dynamic_variable_mapping.items():
                for source, source_info in mapping_info["sources"].items():
                    unit_lookup[source_info["specific_name"]] = source_info["unit"]

        gage_id_lst = self.read_object_ids().tolist()
        total_stations = len(gage_id_lst)
        original_var_lst = self.aqua_fetch.dynamic_features
        cleaned_var_lst = self._clean_feature_names(original_var_lst)
        var_name_mapping = dict(zip(original_var_lst, cleaned_var_lst))

        n_batches = (total_stations + batch_size - 1) // batch_size
        print(
            f"Start batch processing {total_stations} stations, "
            f"{batch_size} per batch ({n_batches} batches)"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        batch_num = 1
        for batch_idx in range(0, total_stations, batch_size):
            batch_end = min(batch_idx + batch_size, total_stations)
            batch_stations = gage_id_lst[batch_idx:batch_end]
            print(
                f"\nBatch {batch_num}/{n_batches} "
                f"(stations {batch_idx}-{batch_end - 1})"
            )

            try:
                # Collect per-station DataFrames via meteo_data_station
                stn_dfs = {}
                for station in batch_stations:
                    df = self.aqua_fetch.meteo_data_station(station)
                    df = df.loc[self.default_t_range[0]:self.default_t_range[1]]
                    stn_dfs[station] = df

                # Build variable-first data_vars then concat along basin dim
                new_data_vars = {}
                for original_var in tqdm(
                    original_var_lst,
                    desc=f"Variables (batch {batch_num})",
                    total=len(original_var_lst),
                ):
                    cleaned_var = var_name_mapping[original_var]
                    var_data = []
                    valid_stations = []
                    for station in batch_stations:
                        df = stn_dfs[station]
                        if original_var in df.columns:
                            da = xr.DataArray(
                                df[original_var].values,
                                dims=["time"],
                                coords={"time": df.index},
                            )
                            var_data.append(da)
                            valid_stations.append(station)

                    if var_data:
                        combined = xr.concat(var_data, dim="basin")
                        combined["basin"] = valid_stations
                        combined.attrs["units"] = unit_lookup.get(
                            cleaned_var, "unknown"
                        )
                        new_data_vars[cleaned_var] = combined

                # Use union of all time coordinates for the batch
                all_times = sorted(
                    set().union(*(df.index for df in stn_dfs.values()))
                )
                batch_ds = xr.Dataset(
                    data_vars=new_data_vars,
                    coords={
                        "basin": batch_stations,
                        "time": all_times,
                    },
                )
                batch_filepath = self.cache_dir.joinpath(
                    f"batch{batch_num:03d}_estreams_timeseries.nc"
                )
                batch_ds.to_netcdf(batch_filepath)
                print(f"Saved batch {batch_num} -> {batch_filepath}")

            except Exception as e:
                print(f"Batch {batch_num} processing failed: {e}")
                import traceback
                traceback.print_exc()
                continue

            batch_num += 1

        print(f"\nAll batches processed! Total {batch_num - 1} batch files saved")

    def read_ts_xrdataset(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        sources: dict = None,
        **kwargs,
    ) -> xr.Dataset:
        """
        Read timeseries data from batch-saved cache files

        Args:
            gage_id_lst: List of station IDs
            t_range: Time range [start, end]
            var_lst: List of standard variable names
            sources: Data source dictionary

        Returns:
            xr.Dataset: xarray dataset containing requested data
        """
        if self._is_cloud():
            # cloud: base class opens the zarr and handles selection/renaming
            return super().read_ts_xrdataset(
                gage_id_lst=gage_id_lst, t_range=t_range,
                var_lst=var_lst, sources=sources, **kwargs,
            )

        if (
            not hasattr(self, "_dynamic_variable_mapping")
            or not self._dynamic_variable_mapping
        ):
            raise NotImplementedError(
                "This dataset does not support the standardized variable mapping."
            )

        if var_lst is None:
            var_lst = list(self._dynamic_variable_mapping.keys())

        if t_range is None:
            t_range = self.default_t_range

        target_vars_to_fetch = []
        rename_map = {}

        # Process variable name mapping and data source selection
        for std_name in var_lst:
            if std_name not in self._dynamic_variable_mapping:
                raise ValueError(
                    f"'{std_name}' is not a recognized standard variable for this dataset."
                )

            mapping_info = self._dynamic_variable_mapping[std_name]

            # Determine which data source(s) to use
            is_explicit_source = sources and std_name in sources
            sources_to_use = []
            if is_explicit_source:
                provided_sources = sources[std_name]
                if isinstance(provided_sources, list):
                    sources_to_use.extend(provided_sources)
                else:
                    sources_to_use.append(provided_sources)
            else:
                sources_to_use.append(mapping_info["default_source"])

            # Only need suffix when user explicitly requests multiple data sources
            needs_suffix = is_explicit_source and len(sources_to_use) > 1
            for source in sources_to_use:
                if source not in mapping_info["sources"]:
                    raise ValueError(
                        f"Source '{source}' is not available for variable '{std_name}'."
                    )

                actual_var_name = mapping_info["sources"][source]["specific_name"]
                target_vars_to_fetch.append(actual_var_name)
                output_name = f"{std_name}_{source}" if needs_suffix else std_name
                rename_map[actual_var_name] = output_name

        # Find all batch files
        batch_pattern = str(self.cache_dir / "batch*_estreams_timeseries.nc")
        batch_files = sorted(glob.glob(batch_pattern))

        if not batch_files:
            print("No batch cache files found, starting cache creation...")
            self.cache_timeseries_xrdataset()
            batch_files = sorted(glob.glob(batch_pattern))

            if not batch_files:
                raise FileNotFoundError("Cache creation failed, no batch files found")

        print(f"Found {len(batch_files)} batch files")

        # If no stations specified, read all stations
        if gage_id_lst is None:
            print("No station list specified, will read all stations...")
            gage_id_lst = self.read_object_ids().tolist()

        # Convert station IDs to strings (ensure consistency)
        gage_id_lst = [str(gid) for gid in gage_id_lst]

        # Iterate through batch files to find batches containing required stations
        relevant_datasets = []
        for batch_file in batch_files:
            try:
                # First open only coordinates, don't load data
                ds_batch = xr.open_dataset(batch_file)
                batch_basins = [str(b) for b in ds_batch.basin.values]

                # Check if this batch contains required stations
                common_basins = list(set(gage_id_lst) & set(batch_basins))

                if common_basins:
                    print(
                        f"Batch {os.path.basename(batch_file)}: contains {len(common_basins)} required stations"
                    )

                    # Check if variables exist
                    missing_vars = [
                        v for v in target_vars_to_fetch if v not in ds_batch.data_vars
                    ]
                    if missing_vars:
                        ds_batch.close()
                        raise ValueError(
                            f"Batch {os.path.basename(batch_file)} missing variables: {missing_vars}"
                        )

                    # Select variables and stations
                    ds_subset = ds_batch[target_vars_to_fetch]
                    ds_selected = ds_subset.sel(
                        basin=common_basins, time=slice(t_range[0], t_range[1])
                    )

                    relevant_datasets.append(ds_selected)
                    ds_batch.close()
                else:
                    ds_batch.close()

            except Exception as e:
                print(f"Failed to read batch file {batch_file}: {e}")
                continue

        if not relevant_datasets:
            raise ValueError(
                f"Specified stations not found in any batch files: {gage_id_lst}"
            )

        print(f"Reading data from {len(relevant_datasets)} batches...")

        # Merge data from all relevant batches
        if len(relevant_datasets) == 1:
            final_ds = relevant_datasets[0]
        else:
            final_ds = xr.concat(relevant_datasets, dim="basin")

        # Rename to standard variable names
        final_ds = final_ds.rename(rename_map)

        # Ensure stations are arranged in input order
        if len(gage_id_lst) > 0:
            # Only select actually existing stations
            existing_basins = [b for b in gage_id_lst if b in final_ds.basin.values]
            if existing_basins:
                final_ds = final_ds.sel(basin=existing_basins)

        return final_ds
