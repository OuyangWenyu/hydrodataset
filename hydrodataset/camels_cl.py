"""
Author: Yimeng Zhang
Date: 2025-10-19 19:40:08
LastEditTime: 2025-10-19 19:40:33
LastEditors: Wenyu Ouyang
Description: CAMELS_CL dataset class
FilePath: \hydrodataset\hydrodataset\camels_cl.py
Copyright (c) 2021-2026 Wenyu Ouyang. All rights reserved.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from aqua_fetch import CAMELS_CL
from hydrodataset import HydroDataset, StandardVariable


class CamelsCl(HydroDataset):
    """CAMELS_CL dataset class extending RainfallRunoff.

    This class provides access to the CAMELS_CL dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    # (folder/file_rel_path, zarr_var_name) –?wide-format TXT per variable
    _FILE_MAP = [
        ("2_CAMELScl_streamflow_m3s/2_CAMELScl_streamflow_m3s.txt",   "q_cms_obs"),
        ("3_CAMELScl_streamflow_mm/3_CAMELScl_streamflow_mm.txt",     "q_mm_obs"),
        ("4_CAMELScl_precip_cr2met/4_CAMELScl_precip_cr2met.txt",     "pcp_mm_cr2met"),
        ("5_CAMELScl_precip_chirps/5_CAMELScl_precip_chirps.txt",     "pcp_mm_chirps"),
        ("6_CAMELScl_precip_mswep/6_CAMELScl_precip_mswep.txt",       "pcp_mm_mswep"),
        ("7_CAMELScl_precip_tmpa/7_CAMELScl_precip_tmpa.txt",         "pcp_mm_tmpa"),
        ("8_CAMELScl_tmin_cr2met/8_CAMELScl_tmin_cr2met.txt",         "airtemp_c_min"),
        ("9_CAMELScl_tmax_cr2met/9_CAMELScl_tmax_cr2met.txt",         "airtemp_c_max"),
        ("10_CAMELScl_tmean_cr2met/10_CAMELScl_tmean_cr2met.txt",     "airtemp_c_mean"),
        ("11_CAMELScl_pet_8d_modis/11_CAMELScl_pet_8d_modis.txt",     "pet_mm_modis"),
        ("12_CAMELScl_pet_hargreaves/12_CAMELScl_pet_hargreaves.txt", "pet_mm_hargreaves"),
        ("13_CAMELScl_swe/13_CAMELScl_swe.txt",                       "swe"),
    ]

    def __init__(
        self, uri: str, region: Optional[str] = None, download: bool = False
    ) -> None:
        super().__init__(uri)
        self.region = region
        self.download = download
        if not str(uri).startswith("s3://"):
            self.aqua_fetch = CAMELS_CL(uri)

    def read_object_ids(self) -> np.ndarray:
        import json
        uri = str(self.data_source_dir).rstrip("/")
        rel = "CAMELS_CL/stations.json"
        if self._is_cloud():
            fs = self._make_s3fs()
            with fs.open(f"{uri}/{rel}".removeprefix("s3://")) as fh:
                ids = json.load(fh)
        else:
            with open(os.path.join(uri, *rel.split("/"))) as fh:
                ids = json.load(fh)
        return np.array(sorted(str(i) for i in ids))

    def cache_attributes_to_zarr(self) -> None:
        import zarr
        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")
        attr_path = f"{uri}/CAMELS_CL/1_CAMELScl_attributes/1_CAMELScl_attributes.txt".removeprefix("s3://")
        with fs.open(attr_path) as fh:
            raw = pd.read_csv(fh, sep="\t", index_col=0, dtype=str)
        # File is transposed: index=attribute names, columns=station IDs
        static = raw.T.copy()
        static.index = static.index.str.strip().astype(str)
        static.columns = self._clean_feature_names(list(static.columns))
        static = static.rename(columns={"area": "area_km2", "gauge_lat": "lat"})
        static = static.apply(pd.to_numeric, errors="ignore")
        # raw attributes only carry per-product p_mean_*; compute the single
        # p_mean from the precipitation timeseries, matching the local cache
        static["p_mean"] = self._p_mean_from_precip(static.index)

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
        base = f"{uri}/CAMELS_CL"

        def _read_wide(file_rel):
            path = f"{base}/{file_rel}".removeprefix("s3://")
            with fs.open(path) as fh:
                df = pd.read_csv(fh, sep="\t", index_col=0, parse_dates=True,
                                 na_values=[" ", ""], dtype=str)
            df.index = pd.to_datetime(df.index)
            df.columns = df.columns.str.strip()
            return df.apply(pd.to_numeric, errors="coerce")

        print("Reading wide-format TXT files from OSS...")
        var_dfs: dict[str, pd.DataFrame] = {}
        for file_rel, zarr_vn in self._FILE_MAP:
            try:
                var_dfs[zarr_vn] = _read_wide(file_rel)
                print(f"  {file_rel.split('/')[-1]} -> {zarr_vn}")
            except Exception as e:
                print(f"  WARN: {file_rel} skipped: {e}")

        stations = self.read_object_ids().tolist()
        all_times = pd.date_range(self.default_t_range[0], self.default_t_range[1], freq="D")
        n, nt = len(stations), len(all_times)
        times_ns = all_times.asi8

        print(f"Writing zarr: {n} stations x {nt} days x {len(var_dfs)} vars")
        zarr_name = self._timeseries_cache_filename.replace(".nc", ".zarr")
        out, opts = self._zarr_path_and_opts(zarr_name)
        root = zarr.open_group(out, mode="w", storage_options=opts, zarr_format=2)

        for zarr_vn, df in var_dfs.items():
            # df is time-indexed with station columns -> (nt, n); transpose to (n, nt)
            data = df.reindex(index=all_times, columns=stations).values.T
            arr = root.create_array(zarr_vn, shape=(n, nt), chunks=(min(n, 100), min(nt, 365)), dtype="float64")
            arr[:] = data
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
        return "camels_cl_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camels_cl_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1913-02-15", "2018-03-09"]

    def cache_attributes_xrdataset(self):
        """Override base method to add calculated p_mean from precipitation timeseries.

        This method:
        1. Calls parent method to create base attribute cache
        2. Reads precipitation timeseries data
        3. Calculates mean precipitation (p_mean) for each basin
        4. Adds p_mean to the attribute dataset
        5. Saves the updated cache
        """
        # Step 1: Create base attribute cache using parent method
        print("Creating base attribute cache...")
        super().cache_attributes_xrdataset()

        # Step 2: Load the base cache file
        cache_file = self.cache_dir.joinpath(self._attributes_cache_filename)
        with xr.open_dataset(cache_file) as ds_attr:
            ds_attr = ds_attr.load()  # Load into memory

        print("Calculating p_mean from precipitation timeseries...")

        # Step 3: Read precipitation timeseries for all basins
        # Use the default precipitation source (cr2met)
        basin_ids = self.read_object_ids().tolist()

        try:
            # Read full precipitation timeseries
            prcp_ts = self.read_ts_xrdataset(
                gage_id_lst=basin_ids,
                t_range=self.default_t_range,
                var_lst=["precipitation"],
            )

            # Step 4: Calculate temporal mean for each basin
            # The result is a DataArray with dimension (basin,)
            p_mean_values = prcp_ts["precipitation"].mean(dim="time")

            # Add units attribute
            p_mean_values.attrs["units"] = "mm/day"
            p_mean_values.attrs["description"] = (
                "Mean daily precipitation (calculated from timeseries)"
            )

            # Step 5: Add p_mean to the attribute dataset
            ds_attr["p_mean"] = p_mean_values

            print(f"Successfully calculated p_mean for {len(basin_ids)} basins")

        except Exception as e:
            print(f"Warning: Could not calculate p_mean from precipitation data: {e}")
            print("Creating p_mean with NaN values as placeholder")
            # Create p_mean with NaN values if calculation fails
            p_mean_nan = xr.DataArray(
                np.full(len(basin_ids), np.nan),
                coords={"basin": basin_ids},
                dims=["basin"],
                attrs={
                    "units": "mm/day",
                    "description": "Mean daily precipitation (not available)",
                },
            )
            ds_attr["p_mean"] = p_mean_nan

        # Step 6: Save the updated cache file
        print(f"Saving updated attribute cache with p_mean to: {cache_file}")
        ds_attr.to_netcdf(cache_file, mode="w")
        print("Successfully saved attribute cache with p_mean")

    # get the information of features from table3 in "https://hess.copernicus.org/articles/22/5817/2018/"
    _subclass_static_definitions = {
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
        "elev_mean": {"specific_name": "elev_mean", "unit": "m"},
        "gauge_lat": {"specific_name": "lat", "unit": "degrees"},
        "gauge_lon": {"specific_name": "long", "unit": "degrees"},
    }
    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "q_cms_obs", "unit": "m^3/s"},
                "depth_based": {"specific_name": "q_mm_obs", "unit": "mm/day"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "cr2met",
            "sources": {
                "cr2met": {"specific_name": "pcp_mm_cr2met", "unit": "mm/day"},
                "chirps": {"specific_name": "pcp_mm_chirps", "unit": "mm/day"},
                "mswep": {"specific_name": "pcp_mm_mswep", "unit": "mm/day"},
                "tmpa": {"specific_name": "pcp_mm_tmpa", "unit": "mm/day"},
            },
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "airtemp_c_min", "unit": "°C"}
            },
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "airtemp_c_max", "unit": "°C"}
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "airtemp_c_mean", "unit": "°C"}
            },
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "modis",
            "sources": {
                "modis": {"specific_name": "pet_mm_modis", "unit": "mm/day"},
                "hargreaves": {"specific_name": "pet_mm_hargreaves", "unit": "mm/day"},
            },
        },
        StandardVariable.SNOW_WATER_EQUIVALENT: {
            "default_source": "observations",
            "sources": {"observations": {"specific_name": "swe", "unit": "mm"}},
        },
    }
