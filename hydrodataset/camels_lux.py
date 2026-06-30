import os
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from hydrodataset import HydroDataset, StandardVariable
from tqdm import tqdm
from aqua_fetch import CAMELS_LUX
from hydroutils import hydro_file


class CamelsLux(HydroDataset):

    _COL_MAP = {
        "Q":        "q_cms_obs",
        "Qspec":    "q_mm_obs",
        "Qflag":    "qflag",
        "RR_rad":   "pcp_mm_radar",
        "RR_min_rad": "rr_min_rad",
        "RR_max_rad": "rr_max_rad",
        "RR_flag_rad": "rr_flag_rad",
        "RR_stn":   "pcp_mm_station",
        "tp":       "pcp_mm_era5",
        "t2m":      "airtemp_c_mean",
        "PET_Oudin":"pet_mm_oudin",
        "PET_PM":   "pet_mm_pm",
        "cape":     "cape",
        "cin":      "cin",
        "kx":       "kx",
        "q":        "spechum_gkg",
        "rh":       "rh_",
        "tcwv":     "tcwv",
        "ws10500":  "windspeed_mps",
        "lls":      "lls",
        "dls":      "dls",
        "swvl1":    "sml1",
        "swvl2":    "sml2",
        "swvl3":    "sml3",
        "swvl4":    "sml4",
    }
    _ATTR_FILES = [
        "CAMELS_LUX_topographic_attributes.csv",
        "CAMELS_LUX_climatic_attributes.csv",
        "CAMELS_LUX_geologic_attributes.csv",
        "CAMELS_LUX_landuse_attributes.csv",
        "CAMELS_LUX_meta_attributes.csv",
    ]
    _DATA_REL = "CAMELS_LUX/CAMELS-LUX"
    """CAMELS_LUX dataset class extending RainfallRunoff.

    This class provides access to the CAMELS_LUX dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    def __init__(
        self, uri: str, region: Optional[str] = None, download: bool = False
    ) -> None:
        """Initialize CAMELS_LUX dataset.

        Args:
            uri: Path to the data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
        """
        super().__init__(uri)
        self.region = region
        self.download = download
        if str(uri).startswith("s3://"):
            return
        try:
            self.aqua_fetch = CAMELS_LUX(uri)
        except Exception as e:
            print(e)
            check_zip_extract = False
            # The zip files that should be downloaded for CAMELS-LUX
            zip_files = ["CAMELS-LUX.zip", "CAMELS-LUX_shapefiles.zip"]
            for filename in tqdm(zip_files, desc="Checking zip files"):
                extracted_dir = self.data_source_dir.joinpath(
                    "CAMELS_LUX", filename[:-4]
                )
                if not extracted_dir.exists():
                    check_zip_extract = True
                    break
            if check_zip_extract:
                hydro_file.zip_extract(self.data_source_dir.joinpath("CAMELS_LUX"))
            self.aqua_fetch = CAMELS_LUX(uri)

    def read_object_ids(self) -> np.ndarray:
        uri = str(self.data_source_dir).rstrip("/")
        ts_rel = f"{self._DATA_REL}/timeseries/daily"
        if self._is_cloud():
            fs = self._make_s3fs()
            names = [p.split("/")[-1] for p in fs.ls(f"{uri}/{ts_rel}".removeprefix("s3://"))]
        else:
            names = os.listdir(os.path.join(uri, *ts_rel.split("/")))
        ids = sorted(
            n.replace("CAMELS_LUX_hydromet_timeseries_", "").replace(".csv", "")
            for n in names if n.startswith("CAMELS_LUX_hydromet_timeseries_")
        )
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
        if "p_mean" not in static.columns:
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
        ts_base = f"{uri}/{self._DATA_REL}/timeseries/daily"

        stations = self.read_object_ids().tolist()
        all_times = pd.date_range(self.default_t_range[0], self.default_t_range[1], freq="D")
        n, nt = len(stations), len(all_times)
        times_ns = all_times.asi8
        all_vars = list(self._COL_MAP.values())

        data: dict[str, np.ndarray] = {vn: np.full((n, nt), np.nan) for vn in all_vars}
        print(f"Reading {n} stations from OSS...")
        for i, stn in enumerate(tqdm(stations, desc="CAMELS_LUX zarr")):
            path = f"{ts_base}/CAMELS_LUX_hydromet_timeseries_{stn}.csv".removeprefix("s3://")
            try:
                with fs.open(path) as fh:
                    df = pd.read_csv(fh, index_col="Date", parse_dates=True)
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
            arr = root.create_array(vn, shape=(n, nt), chunks=(min(n, 56), min(nt, 365)), dtype="float64")
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
        return "camels_lux_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camels_lux_timeseries.nc"

    @property
    def default_t_range(self):
        return ["2004-01-01", "2021-12-31"]

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

    # get the information of features from dataset file"CAMELS-LUX_data-description.pdf"
    _subclass_static_definitions = {
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
        "gauge_lat": {"specific_name": "lat", "unit": "degree"},
        "gauge_lon": {"specific_name": "long", "unit": "degree"},
        "elev_mean": {"specific_name": "elev_mean", "unit": "m"},
        "pet_mean": {"specific_name": "pet_mean", "unit": "mm/day"},
        "Qspec_sum": {"specific_name": "qspec_sum", "unit": "mm/d"},
    }

    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "q_cms_obs", "unit": "m^3/s"},
                "depth_based": {"specific_name": "q_mm_obs", "unit": "mm"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "radar",
            "sources": {
                "radar": {"specific_name": "pcp_mm_radar", "unit": "mm"},
                "station": {"specific_name": "pcp_mm_station", "unit": "mm"},
                "era5": {"specific_name": "pcp_mm_era5", "unit": "mm"},
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": " era5",
            "sources": {
                " era5": {"specific_name": "airtemp_C_mean", "unit": "掳C"},
            },
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "oudin",
            "sources": {
                "oudin": {"specific_name": "pet_mm_oudin", "unit": "mm"},
                "penman_monteith": {"specific_name": "pet_mm_pm", "unit": "mm"},
            },
        },
        StandardVariable.RELATIVE_HUMIDITY: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "rh_", "unit": "%"},
            },
        },
        StandardVariable.SPECIFIC_HUMIDITY: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "spechum_gkg", "unit": "kg/kg"},
            },
        },
        StandardVariable.WIND_SPEED: {
            "default_source": "hersbach",
            "sources": {
                "hersbach": {"specific_name": "windspeed_mps", "unit": "m/s"},
            },
        },
        StandardVariable.LOW_LEVEL_WIND_SHEAR: {
            "default_source": "hersbach",
            "sources": {
                "hersbach": {"specific_name": "lls", "unit": "m/s"},
            },
        },
        StandardVariable.DEEP_LEVEL_WIND_SHEAR: {
            "default_source": "hersbach",
            "sources": {
                "hersbach": {"specific_name": "dls", "unit": "m/s"},
            },
        },
        StandardVariable.CAPE: {
            "default_source": "hersbach",
            "sources": {
                "hersbach": {"specific_name": "cape", "unit": "J/kg"},
            },
        },
        StandardVariable.CIN: {
            "default_source": "hersbach",
            "sources": {
                "hersbach": {"specific_name": "cin", "unit": "J/kg"},
            },
        },
        StandardVariable.MAX_RAIN_RATE: {
            "default_source": "radar",
            "sources": {
                "radar": {"specific_name": "rr_max_rad", "unit": "mm/5Min/1x1km"},
            },
        },
        StandardVariable.MIN_RAIN_RATE: {
            "default_source": "radar",
            "sources": {
                "radar": {"specific_name": "rr_min_rad", "unit": "mm/5Min/1x1km"},
            },
        },  
        StandardVariable.CIN: {
            "default_source": "hersbach",
            "sources": {
                "hersbach": {"specific_name": "cin", "unit": "J/kg"},
            },
        },
        StandardVariable.TOTAL_COLUMN_WATER_VAPOUR: {
            "default_source": "hersbach",
            "sources": {
                "hersbach": {"specific_name": "tcwv", "unit": "J/kg"},
            },
        },
        StandardVariable.TOTAL_COLUMN_WATER_VAPOUR: {
            "default_source": "hersbach",
            "sources": {
                "hersbach": {"specific_name": "tcwv", "unit": "J/kg"},
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER1: {
            "default_source": "Mu帽oz_Sabater",
            "sources": {
                "Mu帽oz_Sabater": {"specific_name": "sml1", "unit": "m^3/m^3"},
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER2: {
            "default_source": "Mu帽oz_Sabater",
            "sources": {
                "Mu帽oz_Sabater": {"specific_name": "sml2", "unit": "m^3/m^3"},
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER3: {
            "default_source": "Mu帽oz_Sabater",
            "sources": {
                "Mu帽oz_Sabater": {"specific_name": "sml3", "unit": "m^3/m^3"},
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER4: {
            "default_source": "Mu帽oz_Sabater",
            "sources": {
                "Mu帽oz_Sabater": {"specific_name": "sml4", "unit": "m^3/m^3"},
            },
        },
    }
