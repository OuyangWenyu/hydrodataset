import os
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from aqua_fetch import CAMELS_AUS
from hydrodataset import HydroDataset, StandardVariable


class CamelsAus(HydroDataset):
    """CAMELS_AUS dataset class."""

    # (raw_file_stem, folder_relative_path, zarr_var_name, scale_factor)
    _FILE_MAP = [
        ("streamflow_MLd",            "03_streamflow/03_streamflow",                                                "q_cms_obs",                  0.01157),
        ("streamflow_MLd_inclInfilled","03_streamflow/03_streamflow",                                               "streamflow_mld_inclinfilled",  1.0),
        ("streamflow_mmd",            "03_streamflow/03_streamflow",                                                "q_mm_obs",                    1.0),
        ("et_morton_actual_SILO",     "05_hydrometeorology/05_hydrometeorology/02_EvaporativeDemand_timeseries",    "aet_mm_silo_morton",           1.0),
        ("et_morton_wet_SILO",        "05_hydrometeorology/05_hydrometeorology/02_EvaporativeDemand_timeseries",    "et_morton_wet_silo",           1.0),
        ("et_morton_point_SILO",      "05_hydrometeorology/05_hydrometeorology/02_EvaporativeDemand_timeseries",    "aet_mm_silo_morton_point",     1.0),
        ("et_short_crop_SILO",        "05_hydrometeorology/05_hydrometeorology/02_EvaporativeDemand_timeseries",    "aet_mm_silo_short_crop",       1.0),
        ("et_tall_crop_SILO",         "05_hydrometeorology/05_hydrometeorology/02_EvaporativeDemand_timeseries",    "aet_mm_silo_tall_crop",        1.0),
        ("evap_morton_lake_SILO",     "05_hydrometeorology/05_hydrometeorology/02_EvaporativeDemand_timeseries",    "evap_morton_lake_silo",        1.0),
        ("evap_pan_SILO",             "05_hydrometeorology/05_hydrometeorology/02_EvaporativeDemand_timeseries",    "evap_pan_silo",                1.0),
        ("evap_syn_SILO",             "05_hydrometeorology/05_hydrometeorology/02_EvaporativeDemand_timeseries",    "evap_syn_silo",                1.0),
        ("precipitation_AGCD",        "05_hydrometeorology/05_hydrometeorology/01_precipitation_timeseries",        "pcp_mm_agcd",                  1.0),
        ("precipitation_SILO",        "05_hydrometeorology/05_hydrometeorology/01_precipitation_timeseries",        "pcp_mm_silo",                  1.0),
        ("precipitation_var_AGCD",    "05_hydrometeorology/05_hydrometeorology/01_precipitation_timeseries",        "precipitation_var_agcd",       1.0),
        ("mslp_SILO",                 "05_hydrometeorology/05_hydrometeorology/03_Other/SILO",                      "mslp_silo",                    1.0),
        ("radiation_SILO",            "05_hydrometeorology/05_hydrometeorology/03_Other/SILO",                      "solrad_wm2_silo",              1.0),
        ("rh_tmax_SILO",              "05_hydrometeorology/05_hydrometeorology/03_Other/SILO",                      "rh__silo_tmax",                1.0),
        ("rh_tmin_SILO",              "05_hydrometeorology/05_hydrometeorology/03_Other/SILO",                      "rh__silo_tmin",                1.0),
        ("tmax_SILO",                 "05_hydrometeorology/05_hydrometeorology/03_Other/SILO",                      "airtemp_c_silo_max",           1.0),
        ("tmin_SILO",                 "05_hydrometeorology/05_hydrometeorology/03_Other/SILO",                      "airtemp_c_silo_min",           1.0),
        ("vp_deficit_SILO",           "05_hydrometeorology/05_hydrometeorology/03_Other/SILO",                      "vp_deficit_silo",              1.0),
        ("vp_SILO",                   "05_hydrometeorology/05_hydrometeorology/03_Other/SILO",                      "vp_hpa_silo",                  1.0),
        ("tmax_AGCD",                 "05_hydrometeorology/05_hydrometeorology/03_Other/AGCD",                      "airtemp_c_agcd_max",           1.0),
        ("tmin_AGCD",                 "05_hydrometeorology/05_hydrometeorology/03_Other/AGCD",                      "airtemp_c_agcd_min",           1.0),
        ("vapourpres_h09_AGCD",       "05_hydrometeorology/05_hydrometeorology/03_Other/AGCD",                      "vp_hpa_agcd_h09",              1.0),
        ("vapourpres_h15_AGCD",       "05_hydrometeorology/05_hydrometeorology/03_Other/AGCD",                      "vp_hpa_agcd_h15",              1.0),
    ]

    def __init__(
        self, uri: str, region: Optional[str] = None, download: bool = False
    ) -> None:
        super().__init__(uri)
        self.region = region
        self.download = download
        if not str(uri).startswith("s3://"):
            self.aqua_fetch = CAMELS_AUS(uri)

    def read_object_ids(self) -> np.ndarray:
        uri = str(self.data_source_dir).rstrip("/")
        csv_rel = "CAMELS_AUS/01_id_name_metadata/01_id_name_metadata/id_name_metadata.csv"
        if self._is_cloud():
            fs = self._make_s3fs()
            with fs.open(f"{uri}/{csv_rel}".removeprefix("s3://")) as fh:
                df = pd.read_csv(fh)
        else:
            df = pd.read_csv(os.path.join(uri, *csv_rel.split("/")))
        return np.array(sorted(df["station_id"].astype(str).tolist()))

    def cache_attributes_to_zarr(self) -> None:
        import zarr
        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")
        csv_path = f"{uri}/CAMELS_AUS/CAMELS_AUS_Attributes&Indices_MasterTable.csv"
        with fs.open(csv_path.removeprefix("s3://")) as fh:
            static = pd.read_csv(fh, index_col="station_id", dtype={"station_id": str})
        static.index = static.index.astype(str)
        static.columns = self._clean_feature_names(list(static.columns))
        static = static.rename(columns={"catchment_area": "area_km2", "lat_outlet": "lat"})

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
        base = f"{uri}/CAMELS_AUS"

        def _read_wide(stem, folder_rel):
            path = f"{base}/{folder_rel}/{stem}.csv".removeprefix("s3://")
            with fs.open(path) as fh:
                df = pd.read_csv(fh, na_values=["-99.99"])
            df["time"] = pd.to_datetime(df[["year", "month", "day"]])
            return df.drop(columns=["year", "month", "day"]).set_index("time")

        print("Reading timeseries CSVs from OSS...")
        var_dfs: dict[str, pd.DataFrame] = {}
        for stem, folder, zarr_name, factor in self._FILE_MAP:
            try:
                df = _read_wide(stem, folder)
                if factor != 1.0:
                    df = df * factor
                var_dfs[zarr_name] = df
                print(f"  {stem} -> {zarr_name}")
            except Exception as e:
                print(f"  WARN: {stem} skipped: {e}")

        # Derived: mean temperature from min/max
        if "airtemp_c_silo_min" in var_dfs and "airtemp_c_silo_max" in var_dfs:
            var_dfs["airtemp_c_mean_silo"] = (var_dfs["airtemp_c_silo_min"] + var_dfs["airtemp_c_silo_max"]) / 2
        if "airtemp_c_agcd_min" in var_dfs and "airtemp_c_agcd_max" in var_dfs:
            var_dfs["airtemp_c_mean_agcd"] = (var_dfs["airtemp_c_agcd_min"] + var_dfs["airtemp_c_agcd_max"]) / 2

        ref_df = var_dfs.get("q_cms_obs", next(iter(var_dfs.values())))
        all_times = ref_df.index.sort_values()
        stations = sorted(ref_df.columns.tolist())
        nt, nb = len(all_times), len(stations)
        times_ns = pd.DatetimeIndex(all_times).asi8

        print(f"Writing zarr: {nb} stations x {nt} timesteps x {len(var_dfs)} vars")
        zarr_name = self._timeseries_cache_filename.replace(".nc", ".zarr")
        out, opts = self._zarr_path_and_opts(zarr_name)
        root = zarr.open_group(out, mode="w", storage_options=opts, zarr_format=2)

        for vn, df in var_dfs.items():
            # df is time-indexed with station columns -> (nt, nb); transpose to (nb, nt)
            data = df.reindex(index=all_times, columns=stations).values.T
            arr = root.create_array(vn, shape=(nb, nt), chunks=(min(nb, 100), min(nt, 365)), dtype="float64")
            arr[:] = data
            arr.attrs["_ARRAY_DIMENSIONS"] = ["basin", "time"]

        time_arr = root.create_array("time", shape=(nt,), chunks=(min(nt, 365),), dtype="int64")
        time_arr[:] = times_ns
        time_arr.attrs["_ARRAY_DIMENSIONS"] = ["time"]
        time_arr.attrs["units"] = "nanoseconds since 1970-01-01"
        time_arr.attrs["calendar"] = "proleptic_gregorian"

        basin_arr = root.create_array("basin", shape=(nb,), chunks=(nb,), dtype=str)
        basin_arr[:] = stations
        basin_arr.attrs["_ARRAY_DIMENSIONS"] = ["basin"]

        root.attrs["coordinates"] = "basin time"
        self._write_zarr_units(root, "dynamic")
        print(f"Timeseries zarr written to: {out}")

    @property
    def _attributes_cache_filename(self):
        return "camels_aus_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camels_aus_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1950-01-01", "2022-03-31"]

    _subclass_static_definitions = {
        "p_mean": {"specific_name": "p_mean", "unit": "mm"},
        "pet_mean": {"specific_name": "pet_mean", "unit": "mm/day"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
        "gauge_lat": {"specific_name": "lat", "unit": "degree"},
        "gauge_lon": {"specific_name": "long", "unit": "degree"},
        "elev_mean": {"specific_name": "elev_mean", "unit": "m"},
        "slope_mean": {"specific_name": "slope_mean", "unit": "m/km"},
        "anngro_mega": {"specific_name": "anngro_mega", "unit": "ML/year"},
    }
    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "bom",
            "sources": {
                "bom": {"specific_name": "q_cms_obs", "unit": "mm^3/s"},
                "gr4j": {
                    "specific_name": "streamflow_mld_inclinfilled",
                    "unit": "ML/day",
                },
                "depth_based": {"specific_name": "q_mm_obs", "unit": "mm/day"},
            },
        },
        StandardVariable.EVAPOTRANSPIRATION: {
            "default_source": "silo_morton",
            "sources": {
                "silo_morton": {
                    "specific_name": "aet_mm_silo_morton",
                    "unit": "mm/day",
                },
            },
        },
        # For PET, AET and ET, the explanation is in the CAMELS_AUS paper, table 2.
        # table 2 in https://essd.copernicus.org/articles/13/3847/2021/#&gid=1&pid=1
        # But the specific names are not the same as the ones in the paper but same as the ones renamed by aqua_fetch.
        # https://github.com/hyex-research/AquaFetch/blob/143c1578fcf18dd6f3a47ba1f2214b089e6e47a9/aqua_fetch/rr/_camels.py#L905C1-L908C93
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "silo_morton",
            "sources": {
                "silo_morton": {
                    "specific_name": "et_morton_wet_silo",
                    "unit": "mm/day",
                },
                "silo_morton_point": {
                    "specific_name": "aet_mm_silo_morton_point",
                    "unit": "mm/day",
                },
                "silo_short_crop": {
                    "specific_name": "aet_mm_silo_short_crop",
                    "unit": "mm/day",
                },
                "silo_tall_crop": {
                    "specific_name": "aet_mm_silo_tall_crop",
                    "unit": "mm/day",
                },
            },
        },
        StandardVariable.EVAPORATION: {
            "default_source": "silo_morton_lake",
            "sources": {
                "silo_morton_lake": {
                    "specific_name": "evap_morton_lake_silo",
                    "unit": "mm/day",
                },
                "silo_pan": {"specific_name": "evap_pan_silo", "unit": "mm/day"},
                "silo_syn": {"specific_name": "evap_syn_silo", "unit": "mm/day"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "agcd",
            "sources": {
                "agcd": {"specific_name": "pcp_mm_agcd", "unit": "mm/day"},
                "silo": {"specific_name": "pcp_mm_silo", "unit": "mm/day"},
                # "agcd_var": {
                #     "specific_name": "precipitation_var_agcd",
                #     "unit": "mm^2/day^2",
                # }, # May not be used
            },
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "agcd",
            "sources": {
                "agcd": {"specific_name": "airtemp_c_agcd_max", "unit": "掳C"},
                "silo": {"specific_name": "airtemp_c_silo_max", "unit": "掳C"},
            },
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "agcd",
            "sources": {
                "agcd": {"specific_name": "airtemp_c_agcd_min", "unit": "掳C"},
                "silo": {"specific_name": "airtemp_c_silo_min", "unit": "掳C"},
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "silo",
            "sources": {
                "silo": {"specific_name": "airtemp_c_mean_silo", "unit": "掳C"},
                "agcd": {"specific_name": "airtemp_c_mean_agcd", "unit": "掳C"},
            },
        },
        StandardVariable.VAPOR_PRESSURE: {
            "default_source": "agcd_h09",
            "sources": {
                "agcd_h09": {"specific_name": "vp_hpa_agcd_h09", "unit": "hPa"},
                "agcd_h15": {"specific_name": "vp_hpa_agcd_h15", "unit": "hPa"},
                "silo": {"specific_name": "vp_hpa_silo", "unit": "hPa"},
                "silo_deficit": {"specific_name": "vp_deficit_silo", "unit": "hPa"},
            },
        },
        StandardVariable.RELATIVE_HUMIDITY: {
            "default_source": "silo",
            "sources": {
                "silo_tmax": {"specific_name": "rh__silo_tmax", "unit": "%"},
                "silo_tmin": {"specific_name": "rh__silo_tmin", "unit": "%"},
            },
        },
        StandardVariable.SURFACE_PRESSURE: {
            "default_source": "silo",
            "sources": {"silo": {"specific_name": "mslp_silo", "unit": "hPa"}},
        },
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "silo",
            "sources": {"silo": {"specific_name": "solrad_wm2_silo", "unit": "MJ/m^2"}},
        },
    }
