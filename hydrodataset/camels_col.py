"""
Author: Yimeng Zhang
Date: 2025-10-19 19:40:08
LastEditTime: 2025-10-19 19:40:19
LastEditors: Wenyu Ouyang
Description: CAMELS_COL dataset class
FilePath: \hydrodataset\hydrodataset\camels_col.py
Copyright (c) 2021-2026 Wenyu Ouyang. All rights reserved.
"""

import io
import os
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from aqua_fetch import CAMELS_COL
from hydrodataset import HydroDataset, StandardVariable
from hydroutils import hydro_file


class CamelsCol(HydroDataset):
    """CAMELS_COL dataset class extending RainfallRunoff.

    This class provides access to the CAMELS_COL dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    # Raw timeseries column 鈫?zarr variable name
    _COL_MAP = {
        "pr":           "pcp_mm",
        "poten_evapo":  "pet_mm",
        "t_max":        "airtemp_c_max",
        "t_min":        "airtemp_c_min",
        "t_mean":       "airtemp_c_mean",
        "streamflow":   "q_cms_obs",
    }
    # Plain static xlsx files (index=gauge_id, read as-is)
    _STATIC_FILES = [
        "02_CAMELS_COL_Catchment_information.xlsx",
        "08_CAMELS_COL_Climatic_indices.xlsx",
        "09_CAMELS_COL_Hydrological_signatures.xlsx",
        "10_CAMELS_COL_Physiograpic_characteristics.xlsx",
    ]
    # AquaFetch static_map: raw column -> standard name
    _STATIC_RENAME = {
        "gauge_lat": "lat",
        "gauge_lon": "long",
        "area": "area_km2",
        "gauge_elev": "elev_gauge_m",
        "perimeter": "perimeter_km",
    }

    def __init__(
        self, uri: str, region: Optional[str] = None, download: bool = False
    ) -> None:
        super().__init__(uri)
        self.region = region
        self.download = download
        if not str(uri).startswith("s3://"):
            self.aqua_fetch = CAMELS_COL(uri)

    def read_object_ids(self) -> np.ndarray:
        uri = str(self.data_source_dir).rstrip("/")
        ts_rel = "CAMELS_COL/04_CAMELS_COL_Hydrometeorological_data/04_CAMELS_COL_Hydrometeorological_data"
        if self._is_cloud():
            fs = self._make_s3fs()
            names = [p.split("/")[-1] for p in fs.ls(f"{uri}/{ts_rel}".removeprefix("s3://"))]
        else:
            names = os.listdir(os.path.join(uri, *ts_rel.split("/")))
        ids = sorted(
            n.replace("Hydromet_data_", "").replace(".txt.txt", "")
            for n in names if n.startswith("Hydromet_data_")
        )
        return np.array(ids)

    def cache_attributes_to_zarr(self) -> None:
        import zarr
        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")

        def _read(fname, **kw):
            path = f"{uri}/CAMELS_COL/{fname}".removeprefix("s3://")
            with fs.open(path, "rb") as fh:
                return pd.read_excel(io.BytesIO(fh.read()), **kw)

        # Plain static files: index = gauge_id
        dfs = []
        for fname in self._STATIC_FILES:
            df = _read(fname, index_col=0, dtype={0: str})
            df.index = df.index.astype(str)
            dfs.append(df)
        static = pd.concat(dfs, axis=1)

        # Geology/landcover/soil are transposed with Catchment_<id> columns.
        # Replicate AquaFetch: geology skips Description/Age/Symbol metadata
        # columns (usecols D:MM); index becomes the gauge id from "Catchment_<id>".
        geol = _read(
            "05_CAMELS_COL_Geologic_characteristics.xlsx",
            index_col=0, dtype={0: str}, usecols="D:MM",
        ).T
        geol.index = [name.split("_")[1] for name in geol.index]
        geol = geol.dropna(axis=1, how="all")

        lc = _read(
            "06_CAMELS_COL_Land_cover_characteristics.xlsx",
            index_col=0, dtype={0: str},
        ).T
        lc.index = [name.split("_")[1] for name in lc.index]
        lc = lc.dropna(axis=1, how="all")

        soil = _read(
            "07_CAMELS_COL_Soil_characteristics.xlsx",
            index_col=0, dtype={0: str},
        ).T
        soil.index = [name.split("_")[1] for name in soil.index]

        static = pd.concat([static, soil, lc, geol], axis=1)
        static = static.rename(columns=self._STATIC_RENAME)

        # AquaFetch converts lat/lon from EPSG:3395 (projected metres) to
        # EPSG:4326 (degrees); replicate so cloud matches the local NC cache.
        R = 6378137.0
        static["long"] = np.degrees(static["long"] / R)
        static["lat"] = np.degrees(
            2 * np.arctan(np.exp(static["lat"] / R)) - np.pi / 2
        )

        static = static.loc[~static.index.duplicated(keep="first")]
        static.columns = self._clean_feature_names(list(static.columns))
        # _clean_feature_names strips non-ASCII chars, which can collapse two
        # distinct geology symbols (e.g. "εO-Sm" and "O-Sm") to the same name;
        # the local NC cache collapses them too, so keep the first occurrence.
        static = static.loc[:, ~static.columns.duplicated(keep="first")]
        if "p_mean" not in static.columns:
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
        print(f"Attributes zarr written to: {out}")

    def cache_timeseries_to_zarr(self) -> None:
        import zarr
        from tqdm import tqdm
        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")
        ts_base = f"{uri}/CAMELS_COL/04_CAMELS_COL_Hydrometeorological_data/04_CAMELS_COL_Hydrometeorological_data"

        stations = self.read_object_ids().tolist()
        all_times = pd.date_range(self.default_t_range[0], self.default_t_range[1], freq="D")
        n, nt = len(stations), len(all_times)
        times_ns = all_times.asi8
        all_vars = list(self._COL_MAP.values())

        data: dict[str, np.ndarray] = {vn: np.full((n, nt), np.nan) for vn in all_vars}
        print(f"Reading {n} stations from OSS...")
        for i, stn in enumerate(tqdm(stations, desc="CAMELS_COL zarr")):
            path = f"{ts_base}/Hydromet_data_{stn}.txt.txt".removeprefix("s3://")
            try:
                with fs.open(path) as fh:
                    df = pd.read_csv(fh, sep="\t", index_col="Date", parse_dates=True)
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
        print(f"Timeseries zarr written to: {out}")

    @property
    def _attributes_cache_filename(self):
        return "camels_col_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camels_col_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1981-05-21", "2022-12-31"]

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
        basin_ids = self.read_object_ids().tolist()

        try:
            # Read full precipitation timeseries
            prcp_ts = self.read_ts_xrdataset(
                gage_id_lst=basin_ids,
                t_range=self.default_t_range,
                var_lst=["precipitation"],
            )

            # Step 4: Calculate temporal mean for each basin
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

    # get the information of features from dataset file "00_CAMELS-COL  Description"
    _subclass_static_definitions = {
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
        "q_mean": {"specific_name": "q_mean", "unit": "m^3/s"},
    }
    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "q_cms_obs", "unit": "m^3/s"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "pcp_mm", "unit": "mm/day"},
            },
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "airtemp_c_max", "unit": "掳C"}
            },
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "airtemp_c_min", "unit": "掳C"},
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "airtemp_c_mean", "unit": "掳C"},
            },
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "pet_mm", "unit": "mm/day"},
            },
        },
    }
