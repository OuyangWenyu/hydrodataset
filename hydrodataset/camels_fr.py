import os
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from hydroutils import hydro_file
from tqdm import tqdm
from hydrodataset import HydroDataset, StandardVariable
from aqua_fetch import CAMELS_FR


class CamelsFr(HydroDataset):

    _COL_MAP = {
        "tsd_q_l":           "q_cms_obs",
        "tsd_q_mm":          "q_mm_obs",
        "tsd_val_s":         "tsd_val_s",
        "tsd_val_q":         "tsd_val_q",
        "tsd_val_m":         "tsd_val_m",
        "tsd_val_c":         "tsd_val_c",
        "tsd_val_i":         "tsd_val_i",
        "tsd_prec":          "pcp_mm",
        "tsd_prec_solid_frac": "pcp_mm_solfrac",
        "tsd_temp":          "airtemp_c_mean",
        "tsd_pet_ou":        "pet_mm_ou",
        "tsd_pet_pe":        "pet_mm_pe",
        "tsd_pet_pm":        "pet_mm_pm",
        "tsd_wind":          "windspeed_mps",
        "tsd_humid":         "spechum_gkg",
        "tsd_rad_dli":       "lwdownrad_wm2",
        "tsd_rad_ssi":       "solrad_wm2",
        "tsd_swi_gr":        "tsd_swi_gr",
        "tsd_swi_isba":      "tsd_swi_isba",
        "tsd_swe_isba":      "tsd_swe_isba",
        "tsd_temp_min":      "airtemp_c_min",
        "tsd_temp_max":      "airtemp_c_max",
    }
    # Only one-row-per-station files (654 rows, sta_code_h3 index). AquaFetch
    # skips the long-format soil/topography quantile files (thousands of rows).
    _ATTR_FILES = [
        "CAMELS_FR_geology_attributes.csv",
        "CAMELS_FR_human_influences_dams.csv",
        "CAMELS_FR_hydrogeology_attributes.csv",
        "CAMELS_FR_land_cover_attributes.csv",
        "CAMELS_FR_station_general_attributes.csv",
        "CAMELS_FR_topography_general_attributes.csv",
    ]
    # AquaFetch static_map: lat/lon from station_general, area from site_general
    _STATIC_RENAME = {
        "sta_y_w84": "lat",
        "sta_x_w84": "long",
        "sit_area_topo": "area_km2",
    }
    _ATTR_REL = "CAMELS_FR/CAMELS_FR_attributes/CAMELS_FR_attributes/static_attributes"
    _TS_REL = "CAMELS_FR/CAMELS_FR_time_series/CAMELS_FR_time_series/daily"
    """CAMELS_FR dataset class extending RainfallRunoff.

    This class provides access to the CAMELS_FR dataset, which contains hourly
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
        """Initialize CAMELS_FR dataset.

        Args:
            uri: Path to the data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
            cache_path: Path to the cache directory
        """
        super().__init__(uri, cache_path=cache_path)
        self.region = region
        self.download = download
        if str(uri).startswith("s3://"):
            return
        try:
            self.aqua_fetch = CAMELS_FR(uri)
        except Exception as e:
            print(e)
            check_zip_extract = False
            # The zip files that should be downloaded for CAMELS-CH
            zip_files = [
                "ADDITIONAL_LICENSES.zip",
                "CAMELS_FR_attributes.zip",
                "CAMELS_FR_geography.zip",
                "CAMELS_FR_time_series.zip",
            ]
            for filename in tqdm(zip_files, desc="Checking zip files"):
                # The extracted directory name (without .zip extension)
                extracted_dir = self.data_source_dir.joinpath(
                    "CAMELS_FR", filename[:-4]
                )
                if not extracted_dir.exists():
                    check_zip_extract = True
                    break
            if check_zip_extract:
                hydro_file.zip_extract(self.data_source_dir.joinpath("CAMELS_FR"))
            self.aqua_fetch = CAMELS_FR(uri)

    def read_object_ids(self) -> np.ndarray:
        uri = str(self.data_source_dir).rstrip("/")
        if self._is_cloud():
            fs = self._make_s3fs()
            names = [p.split("/")[-1] for p in fs.ls(f"{uri}/{self._TS_REL}".removeprefix("s3://"))]
        else:
            names = os.listdir(os.path.join(uri, *self._TS_REL.split("/")))
        ids = sorted(
            n.replace("CAMELS_FR_tsd_", "").replace(".csv", "")
            for n in names if n.startswith("CAMELS_FR_tsd_")
        )
        return np.array(ids)

    def cache_attributes_to_zarr(self) -> None:
        import zarr
        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")
        base = f"{uri}/{self._ATTR_REL}"
        dfs = []
        for fname in self._ATTR_FILES:
            path = f"{base}/{fname}".removeprefix("s3://")
            try:
                with fs.open(path) as fh:
                    df = pd.read_csv(fh, sep=";", index_col="sta_code_h3",
                                     dtype={"sta_code_h3": str})
                df.index = df.index.astype(str)
                dfs.append(df)
            except Exception as e:
                print(f"  WARN {fname}: {e}")
        static = pd.concat(dfs, axis=1)

        # site_general is indexed by sit_code_h3 (8 digits) while the per-station
        # files use sta_code_h3 (10 digits); map by trimming the last 2 digits
        # so sit_area_topo (catchment area) can be joined.
        try:
            site_path = f"{base}/CAMELS_FR_site_general_attributes.csv".removeprefix("s3://")
            with fs.open(site_path) as fh:
                site = pd.read_csv(fh, sep=";", index_col="sit_code_h3",
                                   dtype={"sit_code_h3": str})
            site.index = site.index.astype(str)
            site = site.rename(index={stn[:-2]: stn for stn in static.index})
            static = pd.concat([site, static], axis=1)
        except Exception as e:
            print(f"  WARN site_general join failed: {e}")

        static = static.loc[:, ~static.columns.duplicated(keep="first")]
        static = static.loc[~static.index.duplicated(keep="first")]
        stations = self.read_object_ids().tolist()
        static = static.reindex(stations)
        static.columns = self._clean_feature_names(list(static.columns))
        static = static.rename(columns=self._STATIC_RENAME)
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
        self._write_zarr_units(root, "static")
        print(f"Attributes zarr written to: {out}")

    def cache_timeseries_to_zarr(self) -> None:
        import zarr
        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")
        ts_base = f"{uri}/{self._TS_REL}"

        stations = self.read_object_ids().tolist()
        all_times = pd.date_range(self.default_t_range[0], self.default_t_range[1], freq="D")
        n, nt = len(stations), len(all_times)
        times_ns = all_times.asi8
        all_vars = list(self._COL_MAP.values())

        data: dict[str, np.ndarray] = {vn: np.full((n, nt), np.nan) for vn in all_vars}
        print(f"Reading {n} stations from OSS...")
        for i, stn in enumerate(tqdm(stations, desc="CAMELS_FR zarr")):
            path = f"{ts_base}/CAMELS_FR_tsd_{stn}.csv".removeprefix("s3://")
            try:
                with fs.open(path) as fh:
                    df = pd.read_csv(fh, sep=";", comment="#", dtype=str)
                df.index = pd.to_datetime(df["tsd_date"], format="%Y%m%d")
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
        return "camels_fr_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camels_fr_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1970-01-01", "2021-12-31"]

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

    # get the information of features from dataset file"CAMELS-FR_description"
    _subclass_static_definitions = {
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
        "gauge_lat": {"specific_name": "lat", "unit": "degree"},
        "gauge_lon": {"specific_name": "long", "unit": "degree"},
        "elev_mean": {"specific_name": "elev_mean", "unit": "m"},
        "pet_mean": {"specific_name": "pet_mean", "unit": "mm/day"},
        "hgl_krs_not_karstic": {"specific_name": "hgl_krs_not_karstic", "unit": "-"},
    }

    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "hydroportail",
            "sources": {
                "hydroportail": {"specific_name": "q_cms_obs", "unit": "L/s"},
                "camelsfr": {"specific_name": "q_mm_obs", "unit": "mm/day"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "SIM2-SAFRAN",
            "sources": {
                "SIM2-SAFRAN": {"specific_name": "pcp_mm", "unit": "mm/day"},
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "SIM2-SAFRAN",
            "sources": {
                "SIM2-SAFRAN": {"specific_name": "airtemp_C_mean", "unit": "掳C"},
            },
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "SIM2-SAFRAN",
            "sources": {
                "SIM2-SAFRAN": {"specific_name": "airtemp_C_min", "unit": "掳C"},
            },
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "SIM2-SAFRAN",
            "sources": {
                "SIM2-SAFRAN": {"specific_name": "airtemp_C_max", "unit": "掳C"},
            },
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "oudin",
            "sources": {
                "oudin": {"specific_name": "pet_mm_ou", "unit": "mm/day"},
                "penman": {"specific_name": "pet_mm_pe", "unit": "mm/day"},
                "penman_monteith": {"specific_name": "pet_mm_pm", "unit": "mm/day"},
            },
        },
        StandardVariable.WIND_SPEED: {
            "default_source": "SIM2-SAFRAN",
            "sources": {
                "SIM2-SAFRAN": {"specific_name": "windspeed_mps", "unit": "m/s"},
            },
        },
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "SIM2-SAFRAN",
            "sources": {
                "SIM2-SAFRAN": {"specific_name": "solrad_wm2", "unit": "J/cm^2"},
            },
        },
        StandardVariable.LONGWAVE_SOLAR_RADIATION: {
            "default_source": "SIM2-SAFRAN",
            "sources": {
                "SIM2-SAFRAN": {"specific_name": "lwdownrad_wm2", "unit": "J/cm^2"},
            },
        },
        StandardVariable.SNOW_WATER_EQUIVALENT: {
            "default_source": "isba_model",
            "sources": {
                "isba_model": {"specific_name": "tsd_swe_isba", "unit": "mm/day"},
            },
        },
        StandardVariable.SOIL_MOISTURE: {
            "default_source": "gr",
            "sources": {
                "gr": {"specific_name": "tsd_swi_gr", "unit": "mm/day"},
                "isba": {"specific_name": "tsd_swi_isba", "unit": "mm/day"},
            },
        },      
        StandardVariable.SPECIFIC_HUMIDITY: {
            "default_source": "isba_model",
            "sources": {
                "isba_model": {"specific_name": "spechum_gkg", "unit": "mm/day"},
            },
        },
    }

