import os
import xarray as xr
from typing import Union, List, Optional

from hydrodataset import HydroDataset, StandardVariable
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime
from aqua_fetch import LamaHIce as _AquaFetchLamaHIce
from aqua_fetch.utils import validate_attributes


# Define custom LamaHIce class at module level to avoid pickle issues
# Named LamaHIce to maintain compatibility with file naming conventions
class LamaHIce(_AquaFetchLamaHIce):
    """
    Custom LamaHIce class that overrides URL and methods for new dataset version
    """

    # Override URL at class level
    url = {
        "LamaH-Ice_Caravan_Extension_v15.zip": "https://www.hydroshare.org/resource/705d69c0f77c48538d83cf383f8c63d6/data/contents/LamaH-Ice_Caravan_Extension_v15.zip",
        "lamah_ice.zip": "https://www.hydroshare.org/resource/705d69c0f77c48538d83cf383f8c63d6/data/contents/lamah_ice.zip",
        "lamah_ice_hourly.zip": "https://www.hydroshare.org/resource/705d69c0f77c48538d83cf383f8c63d6/data/contents/lamah_ice_hourly.zip",
    }

    def __init__(
        self,
        path=None,
        overwrite=False,
        *,
        timestep: str = "D",
        data_type: str = "total_upstrm",
        to_netcdf: bool = False,
        **kwargs,
    ):
        """Override __init__ to handle updated URL structure"""
        # don't download hourly data if timestep is daily
        if timestep == "D" and "lamah_ice_hourly.zip" in self.url:
            self.url.pop("lamah_ice_hourly.zip")
        # Updated: changed key from 'Caravan_extension_lamahice.zip' to 'LamaH-Ice_Caravan_Extension_v15.zip'
        if timestep == "H" and "LamaH-Ice_Caravan_Extension_v15.zip" in self.url:
            self.url.pop("LamaH-Ice_Caravan_Extension_v15.zip")

        # Call parent class __init__
        super().__init__(
            path=path,
            timestep=timestep,
            data_type=data_type,
            overwrite=overwrite,
            to_netcdf=to_netcdf,
            **kwargs,
        )

    def fetch_stn_meteo(self, stn: str, nrows: int = None) -> pd.DataFrame:
        """
        Returns climate/meteorological time series data for one station
        Fixed dtype issues for new dataset version
        """
        fpath = os.path.join(self._clim_ts_path(), f"ID_{stn}.csv")

        # Fixed dtypes: changed solar and thermal radiation columns to float32
        dtypes = {
            "YYYY": np.int32,
            "DD": np.int32,
            "MM": np.int32,
            "2m_temp_max": np.float32,
            "2m_temp_mean": np.float32,
            "2m_temp_min": np.float32,
            "2m_dp_temp_max": np.float32,
            "2m_dp_temp_mean": np.float32,
            "2m_dp_temp_min": np.float32,
            "10m_wind_u": np.float32,
            "10m_wind_v": np.float32,
            "fcst_alb": np.float32,
            "lai_high_veg": np.float32,
            "lai_low_veg": np.float32,
            "swe": np.float32,
            "surf_net_solar_rad_max": np.float32,  # Changed from int32 to float32
            "surf_net_solar_rad_mean": np.float32,  # Changed from int32 to float32
            "surf_net_therm_rad_max": np.float32,  # Changed from int32 to float32
            "surf_net_therm_rad_mean": np.float32,  # Changed from int32 to float32
            "surf_press": np.float32,
            "total_et": np.float32,
            "prec": np.float32,
            "volsw_123": np.float32,
            "volsw_4": np.float32,
            "prec_rav": np.float32,
            "prec_carra": np.float32,
        }

        if not os.path.exists(fpath):
            raise FileNotFoundError(f"File not found: {fpath}")

        df = pd.read_csv(fpath, sep=";", dtype=dtypes, nrows=nrows)

        index = df.apply(
            lambda x: datetime.strptime(
                "{0} {1} {2}".format(
                    x["YYYY"].astype(int), x["MM"].astype(int), x["DD"].astype(int)
                ),
                "%Y %m %d",
            ),
            axis=1,
        )

        if self.timestep == "H":
            df.index = index + pd.to_timedelta(df["HOD"], unit="h")
            for col in ["YYYY", "MM", "DD", "DOY", "hh", "mm", "HOD"]:
                df.pop(col)
        else:
            df.index = pd.to_datetime(index)
            for col in [
                "YYYY",
                "MM",
                "DD",
                "DOY",
            ]:
                df.pop(col)

        return df

    def fetch_static_features(
        self,
        stations: Union[str, list] = "all",
        static_features: Union[str, list] = "all",
    ) -> pd.DataFrame:
        """
        Fetches static features of one or more stations with default 'all'
        """
        df = self.static_data()
        df.index = df.index.astype(str)

        static_features = validate_attributes(
            static_features, self.static_features, "static_features"
        )
        stations = validate_attributes(stations, self.stations(), "stations")

        df = df.loc[stations, static_features]
        return df


class LamahIce(HydroDataset):
    """LamaHICE dataset class extending HydroDataset.

    This class uses a custom data reading implementation to support a newer
    dataset version than the one supported by the underlying aquafetch library.
    It overrides the download URLs and provides updated methods.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
    """

    def __init__(
        self, uri: str, region: Optional[str] = None, download: bool = False
    ) -> None:
        """Initialize LamaHICE dataset.

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
        # Use the custom LamaHIce class defined at module level
        self.aqua_fetch = LamaHIce(uri)

    # OSS relative paths (timestep=D, data_type=total_upstrm)
    _P = "LamaHIce/lamah_ice/lamah_ice"
    _BASINS_REL = f"{_P}/A_basins_total_upstrm"
    _CATCH_ATTR_REL = f"{_P}/A_basins_total_upstrm/1_attributes"
    _METEO_REL = f"{_P}/A_basins_total_upstrm/2_timeseries/daily/meteorological_data"
    _GAUGE_ATTR_REL = f"{_P}/D_gauges/1_attributes"
    _Q_REL = f"{_P}/D_gauges/2_timeseries/daily"
    # AquaFetch LamaHIce.static_map
    _STATIC_RENAME = {
        "area_calc_basin": "area_km2",
        "lat_gauge": "lat",
        "slope_mean_basin": "slope_mkm-1",
        "lon_gauge": "long",
    }
    # AquaFetch LamaHIce.dyn_map['D'] resolved to cleaned names
    _DYN_RENAME = {
        "qobs": "q_cms_obs",
        "2m_temp_min": "airtemp_c_2m_min",
        "2m_temp_max": "airtemp_c_2m_max",
        "2m_temp_mean": "airtemp_c_mean_2m",
        "prec": "pcp_mm",
        "pet": "pet_mm",
        "ref_et_rav": "ref_et_mm",
    }

    def read_object_ids(self) -> np.ndarray:
        if self._is_cloud():
            fs = self._make_s3fs()
            uri = str(self.data_source_dir).rstrip("/")
            names = [p.split("/")[-1] for p in fs.ls(f"{uri}/{self._METEO_REL}".removeprefix("s3://"))]
            ids = sorted(
                (n.split(".")[0].split("_")[1] for n in names if n.startswith("ID_")),
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
                df = pd.read_csv(fh, sep=";", index_col="id")
            df.index = df.index.astype(str)
            return df

        # basin attributes = catchment + water_balance(_all) + unfiltered(_unfiltered),
        # all then suffixed with _basin
        cat = _read(self._CATCH_ATTR_REL, "Catchment_attributes.csv")
        wb = _read(self._CATCH_ATTR_REL, "water_balance.csv")
        wb.columns = [c + "_all" for c in wb.columns]
        wbu = _read(self._CATCH_ATTR_REL, "water_balance_unfiltered.csv")
        wbu.columns = [c + "_unfiltered" for c in wbu.columns]
        basin = pd.concat([cat, wb, wbu], axis=1)
        basin.columns = [c + "_basin" for c in basin.columns]

        # gauge attributes = Gauge_attributes + hydro_indices
        g = _read(self._GAUGE_ATTR_REL, "Gauge_attributes.csv")
        hidx = _read(self._GAUGE_ATTR_REL, "hydro_indices_1981_2018.csv")
        gauge = pd.concat([g, hidx], axis=1)

        static = pd.concat([basin, gauge], axis=1)
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

        def _read_dated(path):
            with fs.open(path.removeprefix("s3://")) as fh:
                df = pd.read_csv(fh, sep=";")
            idx = pd.to_datetime(dict(year=df["YYYY"], month=df["MM"], day=df["DD"]))
            df = df.drop(columns=[c for c in ("YYYY", "MM", "DD", "DOY") if c in df.columns])
            df.index = idx
            return df

        data = {vn: np.full((n, nt), np.nan) for vn in cleaned_var_lst}
        for i, stn in enumerate(tqdm(stations, desc="lamah_ice")):
            parts = []
            try:
                parts.append(_read_dated(f"{meteo_base}/ID_{stn}.csv"))
            except Exception as e:
                print(f"  WARN meteo {stn}: {e}")
            try:
                parts.append(_read_dated(f"{q_base}/ID_{stn}.csv"))
            except Exception:
                pass
            if not parts:
                continue
            df = pd.concat(parts, axis=1)
            df = df.loc[~df.index.duplicated(keep="first")]
            df.columns = self._clean_feature_names(
                [self._DYN_RENAME.get(c, c) for c in df.columns]
            )
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
        return "lamahice_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "lamahice_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1950-01-01", "2021-12-31"]

    # Define standardized static variable mappings
    # Based on aqua_fetch LamaHIce static_map
    # information of features get from pdf  https://www.hydroshare.org/resource/705d69c0f77c48538d83cf383f8c63d6/
    _subclass_static_definitions = {
        "p_mean": {"specific_name": "p_mean_basin", "unit": "mm"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
    }

    # Define standardized dynamic variable mappings
    # Based on aqua_fetch LamaHIce dyn_map
    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "lamah_ice",
            "sources": {
                "lamah_ice": {"specific_name": "q_cms_obs", "unit": "m^3/s"},
                "carra": {"specific_name": "runoff_carra", "unit": "mm"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "lamah_ice",
            "sources": {
                "lamah_ice": {"specific_name": "pcp_mm", "unit": "mm"},
                "carra": {"specific_name": "prec_carra", "unit": "mm"},
                "rav": {"specific_name": "prec_rav", "unit": "mm"},
            },
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "lamah_ice",
            "sources": {
                "lamah_ice": {"specific_name": "airtemp_c_2m_min", "unit": "°C"},
                "dp": {"specific_name": "2m_dp_temp_min", "unit": "°C"},
                "carra": {"specific_name": "2m_temp_min_carra", "unit": "°C"},
            },
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "lamah_ice",
            "sources": {
                "lamah_ice": {"specific_name": "airtemp_c_2m_max", "unit": "°C"},
                "dp": {"specific_name": "2m_dp_temp_max", "unit": "°C"},
                "carra": {"specific_name": "2m_temp_max_carra", "unit": "°C"},
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "lamah_ice",
            "sources": {
                "lamah_ice": {"specific_name": "airtemp_c_mean_2m", "unit": "°C"},
                "dp": {"specific_name": "2m_dp_temp_mean", "unit": "°C"},
                "rav": {"specific_name": "2m_temp_rav", "unit": "°C"},
                "carra": {"specific_name": "2m_temp_carra", "unit": "°C"},
            },
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "lamah_ice",
            "sources": {
                "lamah_ice": {"specific_name": "pet_mm", "unit": "mm/day"},
                "caravan": {
                    "specific_name": "potential_evaporation_sum_fao_penman_monteith_from_caravan",
                    "unit": "mm/day",
                },
            },
        },
        StandardVariable.EVAPOTRANSPIRATION: {
            "default_source": "rav",
            "sources": {
                "ref": {"specific_name": "ref_et_mm", "unit": "mm/day"},
                "rav": {"specific_name": "total_et_rav", "unit": "mm/day"},
                "carra": {"specific_name": "total_et_carra", "unit": "mm/day"},
            },
        },
        StandardVariable.U_WIND_SPEED: {
            "default_source": "lamah_ice",
            "sources": {
                "lamah_ice": {"specific_name": "10m_wind_u", "unit": "m/s"},
                "rav": {"specific_name": "10m_wind_u_rav", "unit": "m/s"},
            },
        },
        StandardVariable.V_WIND_SPEED: {
            "default_source": "lamah_ice",
            "sources": {
                "lamah_ice": {"specific_name": "10m_wind_v", "unit": "m/s"},
                "rav": {"specific_name": "10m_wind_v_rav", "unit": "m/s"},
            },
        },
        StandardVariable.WIND_SPEED: {
            "default_source": "carra",
            "sources": {
                "carra": {"specific_name": "10m_wind_speed_carra", "unit": "m/s"}
            },
        },
        StandardVariable.WIND_DIR: {
            "default_source": "carra",
            "sources": {
                "carra": {"specific_name": "10m_wind_dir_carra", "unit": "degree"}
            },
        },
        StandardVariable.SNOW_WATER_EQUIVALENT: {
            "default_source": "lamah_ice",
            "sources": {
                "lamah_ice": {"specific_name": "swe", "unit": "mm"},
                "carra": {"specific_name": "swe_carra", "unit": "mm"},
            },
        },
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "lamah_ice",
            "sources": {
                "lamah_ice": {
                    "specific_name": "surf_net_solar_rad_mean",
                    "unit": "W/m^2",
                },
                "rav": {"specific_name": "surf_dwn_solar_rad_rav", "unit": "W/m^2"},
                "carra": {
                    "specific_name": "surf_net_solar_rad_carra",
                    "unit": "W/m^2",
                },
                "dwn_carra": {
                    "specific_name": "surf_dwn_solar_rad_carra",
                    "unit": "W/m^2",
                },
            },
        },
        StandardVariable.SOLAR_RADIATION_MAX: {
            "default_source": "lamah_ice",
            "sources": {
                "lamah_ice": {
                    "specific_name": "surf_net_solar_rad_max",
                    "unit": "W/m^2",
                }
            },
        },
        StandardVariable.THERMAL_RADIATION: {
            "default_source": "lamah_ice",
            "sources": {
                "lamah_ice": {
                    "specific_name": "surf_net_therm_rad_mean",
                    "unit": "W/m^2",
                },
                "outg": {"specific_name": "surf_outg_therm_rad_rav", "unit": "W/m^2"},
                "dwn": {"specific_name": "surf_dwn_therm_rad_rav", "unit": "W/m^2"},
                "carra": {
                    "specific_name": "surf_net_therm_rad_carra",
                    "unit": "W/m^2",
                },
                "dwn_carra": {
                    "specific_name": "surf_dwn_therm_rad_carra",
                    "unit": "W/m^2",
                },
            },
        },
        StandardVariable.THERMAL_RADIATION_MAX: {
            "default_source": "lamah_ice",
            "sources": {
                "lamah_ice": {
                    "specific_name": "surf_net_therm_rad_max",
                    "unit": "W/m^2",
                }
            },
        },
        StandardVariable.SURFACE_PRESSURE: {
            "default_source": "lamah_ice",
            "sources": {
                "lamah_ice": {"specific_name": "surf_press", "unit": "Pa"},
                "rav": {"specific_name": "surf_press_rav", "unit": "Pa"},
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER1: {
            "default_source": "rav",
            "sources": {"rav": {"specific_name": "volsw_123", "unit": "mm"}},
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER4: {
            "default_source": "rav",
            "sources": {"rav": {"specific_name": "volsw_4", "unit": "mm"}},
        },
        StandardVariable.RELATIVE_HUMIDITY: {
            "default_source": "rav",
            "sources": {
                "rav": {"specific_name": "2m_qv_rav", "unit": "m/s"},
                "carra": {"specific_name": "2m_rel_hum_carra", "unit": "m/s"},
            },
        },
        StandardVariable.SPECIFIC_HUMIDITY: {
            "default_source": "carra",
            "sources": {"carra": {"specific_name": "2m_spec_hum_carra", "unit": "m/s"}},
        },
        StandardVariable.GROUND_HEAT_FLUX: {
            "default_source": "rav",
            "sources": {
                "rav": {"specific_name": "grdflx_rav", "unit": "W/m^2"},
                "sens": {
                    "specific_name": "surf_dwn_sens_heat_flux_carra",
                    "unit": "W/m^2",
                },
                "lat": {
                    "specific_name": "surf_dwn_lat_heat_flux_carra",
                    "unit": "W/m^2",
                },
            },
        },
        StandardVariable.SNOW_SUBLIMATION: {
            "default_source": "carra",
            "sources": {
                "carra": {"specific_name": "snow_sublimation_carra", "unit": "mm"}
            },
        },
        StandardVariable.SOIL_MOISTURE: {
            "default_source": "carra",
            "sources": {"carra": {"specific_name": "percolation_carra", "unit": "mm"}},
        },
    }
