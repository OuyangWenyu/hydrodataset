import os
from typing import Optional

import numpy as np
import pandas as pd

from aqua_fetch import Bull
from hydrodataset import HydroDataset, StandardVariable


class BULL(HydroDataset):
    """Bull dataset class extending RainfallRunoff.

    This class provides access to the Bull dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    def __init__(
        self, uri: str, region: Optional[str] = None, download: bool = False
    ) -> None:
        """Initialize Bull dataset.

        Args:
            uri: Path to the data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
        """
        super().__init__(uri)
        self.region = region
        self.download = download
        if not str(uri).startswith("s3://"):
            self.aqua_fetch = Bull(uri)

    # OSS dataset folder + relative paths (ftype="csv")
    _ATTR_REL = "Bull/attributes"
    _Q_REL = "Bull/timeseries/timeseries/csv/streamflow"
    _AEMET_REL = "Bull/timeseries/timeseries/csv/AEMET"
    _BULL_REL = "Bull/timeseries/timeseries/csv/BULL"
    _ERA5_REL = "Bull/timeseries/timeseries/csv/ERA5_Land"
    _EMO1_REL = "Bull/timeseries/timeseries/csv/EMO1_arc"
    # AquaFetch Bull.static_map (raw -> standard); hydroatlas 'area' pre-renamed
    _STATIC_RENAME = {"area": "area_km2", "gauge_lat": "lat", "gauge_lon": "long"}
    # AquaFetch Bull.dyn_map resolved to cleaned specific names; columns not
    # listed pass through (only cleaned), e.g. surface_pressure_mean_BULL ->
    # surface_pressure_mean_bull, matching the _dynamic_variable_mapping.
    _DYN_RENAME = {
        "dewpoint_temperature_2m_max_BULL": "dptemp_c_max",
        "dewpoint_temperature_2m_mean_BULL": "dptemp_c_mean",
        "dewpoint_temperature_2m_min_BULL": "dptemp_c_min",
        "potential_evaporation_sum_BULL": "pevap_mm",
        "streamflow_BULL": "q_cms_obs",
        "potential_evapotranspiration_AEMET": "pet_mm_aemet",
        "potential_evapotranspiration_EMO1_arc": "pet_mm_emo1arc",
        "potential_evapotranspiration_ERA5_Land": "pet_mm_era5land",
        "surface_net_solar_radiation_mean_BULL": "solrad_wm2",
        "surface_net_solar_radiation_max_BULL": "solrad_wm2_max",
        "surface_net_solar_radiation_min_BULL": "solrad_wm2_min",
        "surface_net_thermal_radiation_max_BULL": "thermrad_wm2_max",
        "surface_net_thermal_radiation_mean_BULL": "thermrad_wm2",
        "surface_net_thermal_radiation_min_BULL": "thermrad_wm2_min",
        "temperature_max_AEMET": "airtemp_c_aemet_max",
        "temperature_max_EMO1_arc": "airtemp_c_emo1arc_max",
        "temperature_max_ERA5_Land": "airtemp_c_era5land_max",
        "temperature_mean_AEMET": "airtemp_c_mean_aemet",
        "temperature_mean_EMO1_arc": "airtemp_c_mean_emo1arc",
        "temperature_mean_ERA5_Land": "airtemp_c_mean_era5land",
        "temperature_min_AEMET": "airtemp_c_aemet_min",
        "temperature_min_EMO1_arc": "airtemp_c_emo1arc_min",
        "temperature_min_ERA5_Land": "airtemp_c_era5land_min",
        "total_precipitation_AEMET": "pcp_mm_aemet",
        "total_precipitation_EMO1_arc": "pcp_mm_emo1arc",
        "total_precipitation_ERA5_Land": "pcp_mm_era5land",
        "total_precipitation_sum_BULL": "pcp_mm_bull",
        "snow_depth_water_equivalent_max_BULL": "swe_mm_max",
        "snow_depth_water_equivalent_mean_BULL": "swe_mm",
        "snow_depth_water_equivalent_min_BULL": "swe_mm_min",
        "temperature_2m_max_BULL": "airtemp_c_2m_max",
        "temperature_2m_mean_BULL": "airtemp_c_mean_2m",
        "temperature_2m_min_BULL": "airtemp_c_2m_min",
        "u_component_of_wind_10m_max_BULL": "windspeedu_mps_max_10m",
        "u_component_of_wind_10m_mean_BULL": "windspeedu_mps_mean_10m",
        "u_component_of_wind_10m_min_BULL": "windspeedu_mps_min_10m",
        "v_component_of_wind_10m_max_BULL": "windspeedv_mps_max_10m",
        "v_component_of_wind_10m_mean_BULL": "windspeedv_mps_mean_10m",
        "v_component_of_wind_10m_min_BULL": "windspeedv_mps_min_10m",
    }
    # (relative dir, filename prefix, column suffix) for the 5 dynamic sources
    _DYN_SOURCES = [
        (_Q_REL, "streamflow", ""),
        (_AEMET_REL, "AEMET", "_AEMET"),
        (_BULL_REL, "BULL", "_BULL"),
        (_ERA5_REL, "ERA5_Land", "_ERA5_Land"),
        (_EMO1_REL, "EMO1", "_EMO1_arc"),
    ]

    def read_object_ids(self) -> np.ndarray:
        if self._is_cloud():
            fs = self._make_s3fs()
            uri = str(self.data_source_dir).rstrip("/")
            names = [p.split("/")[-1] for p in fs.ls(f"{uri}/{self._Q_REL}".removeprefix("s3://"))]
            ids = sorted(
                "BULL_" + n.split(".")[0].split("_")[1]
                for n in names if n.startswith("streamflow_")
            )
            return np.array(ids)
        return super().read_object_ids()

    def cache_attributes_to_zarr(self) -> None:
        import zarr

        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")
        base = f"{uri}/{self._ATTR_REL}"

        def _read(fname):
            with fs.open(f"{base}/{fname}".removeprefix("s3://")) as fh:
                df = pd.read_csv(fh, index_col=0)
            df.index = df.index.astype(str)
            return df

        caravan = _read("attributes_caravan_.csv")
        hydro = _read("attributes_hydroatlas_.csv").rename(columns={"area": "area_hydroatlas"})
        other = _read("attributes_other_ss.csv")
        static = pd.concat([caravan, hydro, other], axis=1)
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

    def cache_timeseries_to_zarr(self, batch_size: int = 50) -> None:
        import zarr
        from tqdm import tqdm

        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")

        stations = self.read_object_ids().tolist()
        n = len(stations)
        all_times = pd.date_range(self.default_t_range[0], self.default_t_range[1], freq="D")
        nt = len(all_times)
        times_ns = all_times.asi8

        # all source-specific names from the dynamic mapping
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

        def _read_src(rel, prefix, suffix, numeric_id):
            path = f"{uri}/{rel}/{prefix}_{numeric_id}.csv".removeprefix("s3://")
            try:
                with fs.open(path) as fh:
                    df = pd.read_csv(fh, index_col="date", parse_dates=True)
            except Exception:
                return None
            if suffix:
                df.columns = [f"{c}{suffix}" for c in df.columns]
            return df

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
                numeric_id = stn.split("_")[1]
                parts = []
                for rel, prefix, suffix in self._DYN_SOURCES:
                    d = _read_src(rel, prefix, suffix, numeric_id)
                    if d is not None:
                        parts.append(d)
                if not parts:
                    continue
                df = pd.concat(parts, axis=1)
                df.index = pd.to_datetime(df.index)
                df = df[~df.index.duplicated(keep="first")]
                df.columns = self._clean_feature_names(
                    [self._DYN_RENAME.get(c, c) for c in df.columns]
                )
                df = df.reindex(all_times)
                for vn in cleaned_var_lst:
                    if vn in df.columns:
                        buffers[vn][j] = pd.to_numeric(df[vn], errors="coerce").values
            for vn in cleaned_var_lst:
                root[vn][start:end, :] = buffers[vn]
            progress[start:end] = 1
            print(f"Batch {bnum}/{n_batches}: done")
        self._write_zarr_units(root, "dynamic")
        print(f"Timeseries zarr written to: {out}")

    @property
    def _attributes_cache_filename(self):
        return "bull_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "bull_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1951-01-02", "2021-12-31"]

    _subclass_static_definitions = {
        "area": {"specific_name": "area_km2", "unit": "km^2"},
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
    }

    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "streamflow", "unit": "m^3/s"},
                "q_cms": {"specific_name": "q_cms_obs", "unit": "m^3/s"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "aemet",
            "sources": {
                "aemet": {"specific_name": "pcp_mm_aemet", "unit": "mm/day"},
                "bull": {"specific_name": "pcp_mm_bull", "unit": "mm/day"},
                "era5land": {"specific_name": "pcp_mm_era5land", "unit": "mm/day"},
                "emo1arc": {"specific_name": "pcp_mm_emo1arc", "unit": "mm/day"},
            },
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "aemet",
            "sources": {
                "aemet": {"specific_name": "airtemp_c_aemet_max", "unit": "°C"},
                "era5land": {"specific_name": "airtemp_c_era5land_max", "unit": "°C"},
                "emo1arc": {"specific_name": "airtemp_c_emo1arc_max", "unit": "°C"},
                "2m": {"specific_name": "airtemp_c_2m_max", "unit": "°C"},
                "dewpoint": {"specific_name": "dptemp_c_max", "unit": "°C"},
            },
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "aemet",
            "sources": {
                "aemet": {"specific_name": "airtemp_c_aemet_min", "unit": "°C"},
                "era5land": {"specific_name": "airtemp_c_era5land_min", "unit": "°C"},
                "emo1arc": {"specific_name": "airtemp_c_emo1arc_min", "unit": "°C"},
                "2m": {"specific_name": "airtemp_c_2m_min", "unit": "°C"},
                "dewpoint": {"specific_name": "dptemp_c_min", "unit": "°C"},
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "aemet",
            "sources": {
                "aemet": {"specific_name": "airtemp_c_mean_aemet", "unit": "°C"},
                "era5land": {"specific_name": "airtemp_c_mean_era5land", "unit": "°C"},
                "emo1arc": {"specific_name": "airtemp_c_mean_emo1arc", "unit": "°C"},
                "2m": {"specific_name": "airtemp_c_mean_2m", "unit": "°C"},
                "dewpoint": {"specific_name": "dptemp_c_mean", "unit": "°C"},
            },
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "aemet",
            "sources": {
                "aemet": {"specific_name": "pet_mm_aemet", "unit": "mm/day"},
                "era5land": {"specific_name": "pet_mm_era5land", "unit": "mm/day"},
                "emo1arc": {"specific_name": "pet_mm_emo1arc", "unit": "mm/day"},
            },
        },
        StandardVariable.EVAPORATION: {
            "default_source": "bull",
            "sources": {"bull": {"specific_name": "pevap_mm", "unit": "mm/day"}},
        },
        # Snow water equivalent - separate MIN and MAX as independent variables
        StandardVariable.SNOW_WATER_EQUIVALENT: {
            "default_source": "observations",
            "sources": {"observations": {"specific_name": "swe_mm", "unit": "mm"}},
        },
        StandardVariable.SNOW_WATER_EQUIVALENT_MIN: {
            "default_source": "observations",
            "sources": {"observations": {"specific_name": "swe_mm_min", "unit": "mm"}},
        },
        StandardVariable.SNOW_WATER_EQUIVALENT_MAX: {
            "default_source": "observations",
            "sources": {"observations": {"specific_name": "swe_mm_max", "unit": "mm"}},
        },
        # Solar radiation - separate MIN and MAX as independent variables
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "solrad_wm2", "unit": "W/m^2"}
            },
        },
        StandardVariable.SOLAR_RADIATION_MIN: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "solrad_wm2_min", "unit": "W/m^2"}
            },
        },
        StandardVariable.SOLAR_RADIATION_MAX: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "solrad_wm2_max", "unit": "W/m^2"}
            },
        },
        # Thermal radiation - separate MIN and MAX as independent variables
        StandardVariable.THERMAL_RADIATION: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "thermrad_wm2", "unit": "W/m^2"}
            },
        },
        StandardVariable.THERMAL_RADIATION_MIN: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "thermrad_wm2_min", "unit": "W/m^2"}
            },
        },
        StandardVariable.THERMAL_RADIATION_MAX: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "thermrad_wm2_max", "unit": "W/m^2"}
            },
        },
        # Surface pressure - separate MIN and MAX as independent variables
        StandardVariable.SURFACE_PRESSURE: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "surface_pressure_mean_bull",
                    "unit": "Pa",
                }
            },
        },
        StandardVariable.SURFACE_PRESSURE_MIN: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "surface_pressure_min_bull",
                    "unit": "Pa",
                }
            },
        },
        StandardVariable.SURFACE_PRESSURE_MAX: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "surface_pressure_max_bull",
                    "unit": "Pa",
                }
            },
        },
        # U wind speed - separate MIN and MAX as independent variables
        StandardVariable.U_WIND_SPEED: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "windspeedu_mps_mean_10m",
                    "unit": "m/s",
                }
            },
        },
        StandardVariable.U_WIND_SPEED_MIN: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "windspeedu_mps_min_10m",
                    "unit": "m/s",
                }
            },
        },
        StandardVariable.U_WIND_SPEED_MAX: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "windspeedu_mps_max_10m",
                    "unit": "m/s",
                }
            },
        },
        # V wind speed - separate MIN and MAX as independent variables
        StandardVariable.V_WIND_SPEED: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "windspeedv_mps_mean_10m",
                    "unit": "m/s",
                }
            },
        },
        StandardVariable.V_WIND_SPEED_MIN: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "windspeedv_mps_min_10m",
                    "unit": "m/s",
                }
            },
        },
        StandardVariable.V_WIND_SPEED_MAX: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "windspeedv_mps_max_10m",
                    "unit": "m/s",
                }
            },
        },
        # Volumetric soil water layer 1 - separate MIN and MAX as independent variables
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER1: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "volumetric_soil_water_layer_1_mean_bull",
                    "unit": "m^3/m^3",
                }
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER1_MIN: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "volumetric_soil_water_layer_1_min_bull",
                    "unit": "m^3/m^3",
                }
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER1_MAX: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "volumetric_soil_water_layer_1_max_bull",
                    "unit": "m^3/m^3",
                }
            },
        },
        # Volumetric soil water layer 2 - separate MIN and MAX as independent variables
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER2: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "volumetric_soil_water_layer_2_mean_bull",
                    "unit": "m^3/m^3",
                }
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER2_MIN: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "volumetric_soil_water_layer_2_min_bull",
                    "unit": "m^3/m^3",
                }
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER2_MAX: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "volumetric_soil_water_layer_2_max_bull",
                    "unit": "m^3/m^3",
                }
            },
        },
        # Volumetric soil water layer 3 - separate MIN and MAX as independent variables
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER3: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "volumetric_soil_water_layer_3_mean_bull",
                    "unit": "m^3/m^3",
                }
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER3_MIN: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "volumetric_soil_water_layer_3_min_bull",
                    "unit": "m^3/m^3",
                }
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER3_MAX: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "volumetric_soil_water_layer_3_max_bull",
                    "unit": "m^3/m^3",
                }
            },
        },
        # Volumetric soil water layer 4 - separate MIN and MAX as independent variables
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER4: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "volumetric_soil_water_layer_4_mean_bull",
                    "unit": "m^3/m^3",
                }
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER4_MIN: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "volumetric_soil_water_layer_4_min_bull",
                    "unit": "m^3/m^3",
                }
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER4_MAX: {
            "default_source": "observations",
            "sources": {
                "observations": {
                    "specific_name": "volumetric_soil_water_layer_4_max_bull",
                    "unit": "m^3/m^3",
                }
            },
        },
    }
