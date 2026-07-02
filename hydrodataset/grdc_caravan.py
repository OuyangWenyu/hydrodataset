import os
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from aqua_fetch import GRDCCaravan as _AquaFetchGRDCCaravan
from hydrodataset import HydroDataset, StandardVariable


# Define the custom GRDCCaravan class at module level to avoid pickle errors
# Named GRDCCaravan to maintain compatibility with file naming conventions
class GRDCCaravan(_AquaFetchGRDCCaravan):
    """Custom GRDCCaravan class with updated URLs and paths for the new dataset version."""

    # Updated URLs for the new dataset version
    url = {
        "GRDC_Caravan_extension_csv.zip": "https://zenodo.org/records/15349031/files/GRDC_Caravan_extension_csv.zip?download=1",
        "GRDC_Caravan_extension_nc.zip": "https://zenodo.org/records/15349031/files/GRDC_Caravan_extension_nc.zip?download=1",
        "grdc-caravan_data_description.pdf": "https://zenodo.org/records/15349031/files/grdc-caravan_data_description.pdf?download=1",
    }

    def __init__(
        self, path=None, overwrite: bool = False, verbosity: int = 1, **kwargs
    ):
        """Custom initialization that uses new URLs and file names."""
        # Import necessary modules
        from aqua_fetch._backend import xarray as xr_backend

        if xr_backend is None:
            self.ftype = "csv"
            if "GRDC_Caravan_extension_nc.zip" in self.url:
                self.url.pop("GRDC_Caravan_extension_nc.zip")
        else:
            self.ftype = "netcdf"
            if "GRDC_Caravan_extension_csv.zip" in self.url:
                self.url.pop("GRDC_Caravan_extension_csv.zip")

        # Call the grandparent class init (from _RainfallRunoff) directly
        from aqua_fetch.rr.utils import _RainfallRunoff

        _RainfallRunoff.__init__(self, path=path, verbosity=verbosity, **kwargs)

        if not os.path.exists(self.path):
            if self.verbosity > 1:
                print(f"Creating directory {self.path}")
            os.makedirs(self.path)

        from aqua_fetch.utils import download, unzip

        for _file, url in self.url.items():
            fpath = os.path.join(self.path, _file)
            if not os.path.exists(fpath) and not overwrite:
                if self.verbosity > 0:
                    print(f"Downloading {_file} from {url}")
                download(url, outdir=self.path, fname=_file)
                unzip(self.path)
            elif self.verbosity > 0:
                print(f"{_file} at {self.path} already exists")

        # Cache stations and attributes
        self._stations = self.other_attributes().index.to_list()
        self._static_attributes = self._static_data().columns.tolist()
        self._dynamic_attributes = self._read_stn_dyn(
            self.stations()[0]
        ).columns.tolist()

    @property
    def shapefiles_path(self):
        """Custom shapefiles_path with updated directory names."""
        if self.ftype == "csv":
            return os.path.join(
                self.path,
                "GRDC_Caravan_extension_csv",
                "GRDC_Caravan_extension_csv",
                "shapefiles",
                "grdc",
            )
        return os.path.join(
            self.path,
            "GRDC_Caravan_extension_nc",
            "GRDC_Caravan_extension_nc",
            "shapefiles",
            "grdc",
        )

    @property
    def attrs_path(self):
        """Custom attrs_path with updated directory names."""
        if self.ftype == "csv":
            return os.path.join(
                self.path,
                "GRDC_Caravan_extension_csv",
                "GRDC_Caravan_extension_csv",
                "attributes",
                "grdc",
            )
        return os.path.join(
            self.path,
            "GRDC_Caravan_extension_nc",
            "GRDC_Caravan_extension_nc",
            "attributes",
            "grdc",
        )

    @property
    def ts_path(self):
        """Custom ts_path with updated directory names."""
        if self.ftype == "csv":
            return os.path.join(
                self.path,
                "GRDC_Caravan_extension_csv",
                "GRDC_Caravan_extension_csv",
                "timeseries",
                "grdc",
            )

        return os.path.join(
            self.path,
            "GRDC_Caravan_extension_nc",
            "GRDC_Caravan_extension_nc",
            "timeseries",
            self.ftype,
            "grdc",
        )


class GrdcCaravan(HydroDataset):
    """GRDC-Caravan dataset class extending HydroDataset.

    This class provides access to the GRDC-Caravan dataset, which contains
    hydrological and meteorological data for watersheds globally.

    This class uses a custom data reading implementation to support a newer
    dataset version than the one supported by the underlying aquafetch library.
    It overrides the download URLs and provides its own parsing and caching logic.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
    """

    def __init__(
        self,
        uri: str,
        region: Optional[str] = None,
        download: bool = False,
        cache_path: Optional[str] = None,
    ) -> None:
        """Initialize GRDC-Caravan dataset.

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
        # Instantiate the custom class defined at module level
        self.aqua_fetch = GRDCCaravan(uri)

    # OSS relative paths (uploaded folder GRDCCaravan; nc extension is populated)
    _EXT = "GRDCCaravan/GRDC_Caravan_extension_nc/GRDC_Caravan_extension_nc"
    _ATTR_REL = f"{_EXT}/attributes/grdc"
    _TS_REL = f"{_EXT}/timeseries/netcdf/grdc"
    # AquaFetch GRDCCaravan.static_map
    _STATIC_RENAME = {"area": "area_km2", "gauge_lat": "lat", "gauge_lon": "long"}
    # AquaFetch GRDCCaravan.dyn_map resolved to cleaned names (others pass through)
    _DYN_RENAME = {
        "streamflow": "q_mm_obs",
        "temperature_2m_mean": "airtemp_c_mean_2m",
        "temperature_2m_min": "airtemp_c_2m_min",
        "temperature_2m_max": "airtemp_c_2m_max",
        "total_precipitation_sum": "pcp_mm",
    }

    def read_object_ids(self) -> np.ndarray:
        if self._is_cloud():
            fs = self._make_s3fs()
            uri = str(self.data_source_dir).rstrip("/")
            names = [p.split("/")[-1] for p in fs.ls(f"{uri}/{self._TS_REL}".removeprefix("s3://"))]
            ids = sorted(n[:-3] for n in names if n.endswith(".nc"))
            return np.array(ids)
        return super().read_object_ids()

    def cache_attributes_to_zarr(self) -> None:
        import zarr

        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")
        base = f"{uri}/{self._ATTR_REL}"

        def _read(fname):
            with fs.open(f"{base}/{fname}".removeprefix("s3://")) as fh:
                df = pd.read_csv(fh, index_col="gauge_id")
            df.index = df.index.astype(str)
            return df

        other = _read("attributes_other_grdc.csv")
        hydro = _read("attributes_hydroatlas_grdc.csv")
        caravan = _read("attributes_caravan_grdc.csv")
        static = pd.concat([other, hydro, caravan], axis=1)
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

    def cache_timeseries_to_zarr(self, batch_size: int = 200) -> None:
        import zarr
        import netCDF4 as nc4

        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")
        ts_base = f"{uri}/{self._TS_REL}"

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
                path = f"{ts_base}/{stn}.nc".removeprefix("s3://")
                try:
                    buf = fs.cat(path)
                    ds = xr.open_dataset(xr.backends.NetCDF4DataStore(
                        nc4.Dataset("inmem", memory=buf)))
                    df = ds.to_dataframe()
                    ds.close()
                    if "date" in df.index.names:
                        df.index = pd.to_datetime(df.index.get_level_values("date"))
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
        return "grdc_caravan_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "grdc_caravan_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1950-01-02", "2023-05-18"]

    # get the information of features from grdc-caravan_data_description.pdf
    # Static variable definitions based on inspected data
    _subclass_static_definitions = {
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
    }

    # Dynamic variable mapping based on inspected data
    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "mm",
            "sources": {
                "mm": {"specific_name": "q_mm_obs", "unit": "mm/day"},
                "cms": {"specific_name": "q_cms_obs", "unit": "m^3/s"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "pcp_mm", "unit": "mm/day"},
            },
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "airtemp_c_2m_max", "unit": "°C"},
                "dewpoint": {
                    "specific_name": "dewpoint_temperature_2m_max",
                    "unit": "°C",
                },
            },
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "airtemp_c_2m_min", "unit": "°C"},
                "dewpoint": {
                    "specific_name": "dewpoint_temperature_2m_min",
                    "unit": "°C",
                },
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "airtemp_c_mean_2m", "unit": "°C"},
                "dewpoint": {
                    "specific_name": "dewpoint_temperature_2m_mean",
                    "unit": "°C",
                },
            },
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "era5_land",
            "sources": {
                "era5_land": {
                    "specific_name": "potential_evaporation_sum_era5_land",
                    "unit": "mm/day",
                },
                "fao_pm": {
                    "specific_name": "potential_evaporation_sum_fao_penman_monteith",
                    "unit": "mm/day",
                },
            },
        },
        StandardVariable.SNOW_WATER_EQUIVALENT_MAX: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "snow_depth_water_equivalent_max",
                    "unit": "m",
                },
            },
        },
        StandardVariable.SNOW_WATER_EQUIVALENT_MIN: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "snow_depth_water_equivalent_min",
                    "unit": "m",
                },
            },
        },
        StandardVariable.SNOW_WATER_EQUIVALENT: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "snow_depth_water_equivalent_mean",
                    "unit": "m",
                },
            },
        },
        StandardVariable.SOLAR_RADIATION_MAX: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "surface_net_solar_radiation_max",
                    "unit": "W/m^2",
                },
            },
        },
        StandardVariable.SOLAR_RADIATION_MIN: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "surface_net_solar_radiation_min",
                    "unit": "W/m^2",
                },
            },
        },
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "surface_net_solar_radiation_mean",
                    "unit": "W/m^2",
                },
            },
        },
        StandardVariable.THERMAL_RADIATION_MAX: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "surface_net_thermal_radiation_max",
                    "unit": "W/m^2",
                },
            },
        },
        StandardVariable.THERMAL_RADIATION_MIN: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "surface_net_thermal_radiation_min",
                    "unit": "W/m^2",
                },
            },
        },
        StandardVariable.THERMAL_RADIATION: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "surface_net_thermal_radiation_mean",
                    "unit": "W/m^2",
                },
            },
        },
        StandardVariable.SURFACE_PRESSURE_MAX: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "surface_pressure_max", "unit": "Pa"},
            },
        },
        StandardVariable.SURFACE_PRESSURE_MIN: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "surface_pressure_min", "unit": "Pa"},
            },
        },
        StandardVariable.SURFACE_PRESSURE: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "surface_pressure_mean", "unit": "Pa"},
            },
        },
        StandardVariable.U_WIND_SPEED_MAX: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "u_component_of_wind_10m_max", "unit": "m/s"},
            },
        },
        StandardVariable.U_WIND_SPEED_MIN: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "u_component_of_wind_10m_min", "unit": "m/s"},
            },
        },
        StandardVariable.U_WIND_SPEED: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "u_component_of_wind_10m_mean",
                    "unit": "m/s",
                },
            },
        },
        StandardVariable.V_WIND_SPEED_MAX: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "v_component_of_wind_10m_max", "unit": "m/s"},
            },
        },
        StandardVariable.V_WIND_SPEED_MIN: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "v_component_of_wind_10m_min", "unit": "m/s"},
            },
        },
        StandardVariable.V_WIND_SPEED: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "v_component_of_wind_10m_mean",
                    "unit": "m/s",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER1_MAX: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "volumetric_soil_water_layer_1_max",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER1_MIN: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "volumetric_soil_water_layer_1_min",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER1: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "volumetric_soil_water_layer_1_mean",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER2_MAX: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "volumetric_soil_water_layer_2_max",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER2_MIN: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "volumetric_soil_water_layer_2_min",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER2: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "volumetric_soil_water_layer_2_mean",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER3_MAX: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "volumetric_soil_water_layer_3_max",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER3_MIN: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "volumetric_soil_water_layer_3_min",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER3: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "volumetric_soil_water_layer_3_mean",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER4_MAX: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "volumetric_soil_water_layer_4_max",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER4_MIN: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "volumetric_soil_water_layer_4_min",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER4: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "volumetric_soil_water_layer_4_mean",
                    "unit": "m^3/m^3",
                },
            },
        },
    }

    def cache_timeseries_xrdataset(self, batch_size=200):
        """Cache timeseries to NetCDF in batches, one file per batch.

        GRDC-Caravan has thousands of stations; fetching them all at once
        exhausts memory, so we process ``batch_size`` stations at a time and
        write each batch as ``batch{N}_grdc_caravan_timeseries.nc``.
        """
        if not hasattr(self, "aqua_fetch"):
            raise NotImplementedError("aqua_fetch attribute is required")

        # Build mapping from specific variable names to units
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
            f"{batch_size} stations per batch ({n_batches} batches)"
        )

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        batch_num = 1
        for batch_idx in range(0, total_stations, batch_size):
            batch_end = min(batch_idx + batch_size, total_stations)
            batch_stations = gage_id_lst[batch_idx:batch_end]
            print(
                f"\nProcessing batch {batch_num}/{n_batches} "
                f"(stations {batch_idx}-{batch_end - 1})"
            )

            try:
                batch_data = self.aqua_fetch.fetch_stations_features(
                    stations=batch_stations,
                    dynamic_features=original_var_lst,
                    static_features=None,
                    st=self.default_t_range[0],
                    en=self.default_t_range[1],
                    as_dataframe=False,
                )
                dynamic_data = (
                    batch_data[1] if isinstance(batch_data, tuple) else batch_data
                )

                new_data_vars = {}
                time_coord = dynamic_data.coords["time"]
                for original_var in tqdm(
                    original_var_lst,
                    desc=f"Variables (batch {batch_num})",
                    total=len(original_var_lst),
                ):
                    cleaned_var = var_name_mapping[original_var]
                    var_data = []
                    for station in batch_stations:
                        if station in dynamic_data.data_vars:
                            station_data = dynamic_data[station].sel(
                                dynamic_features=original_var
                            )
                            if "dynamic_features" in station_data.coords:
                                station_data = station_data.drop("dynamic_features")
                            var_data.append(station_data)

                    if var_data:
                        combined = xr.concat(var_data, dim="basin")
                        combined["basin"] = batch_stations
                        combined.attrs["units"] = unit_lookup.get(
                            cleaned_var, "unknown"
                        )
                        new_data_vars[cleaned_var] = combined

                batch_ds = xr.Dataset(
                    data_vars=new_data_vars,
                    coords={"basin": batch_stations, "time": time_coord},
                )
                batch_filepath = self.cache_dir.joinpath(
                    f"batch{batch_num:03d}_grdc_caravan_timeseries.nc"
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
        """Read timeseries from the batch-saved cache (standard names + sources)."""
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
        for std_name in var_lst:
            if std_name not in self._dynamic_variable_mapping:
                raise ValueError(
                    f"'{std_name}' is not a recognized standard variable for this dataset."
                )
            mapping_info = self._dynamic_variable_mapping[std_name]
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

        import glob

        batch_pattern = str(self.cache_dir / "batch*_grdc_caravan_timeseries.nc")
        batch_files = sorted(glob.glob(batch_pattern))
        if not batch_files:
            print("No batch cache files found, starting cache creation...")
            self.cache_timeseries_xrdataset()
            batch_files = sorted(glob.glob(batch_pattern))
            if not batch_files:
                raise FileNotFoundError("Cache creation failed, no batch files found")

        if gage_id_lst is None:
            gage_id_lst = self.read_object_ids().tolist()
        gage_id_lst = [str(gid) for gid in gage_id_lst]

        relevant_datasets = []
        for batch_file in batch_files:
            try:
                ds_batch = xr.open_dataset(batch_file)
                batch_basins = [str(b) for b in ds_batch.basin.values]
                common_basins = list(set(gage_id_lst) & set(batch_basins))
                if common_basins:
                    missing_vars = [
                        v for v in target_vars_to_fetch if v not in ds_batch.data_vars
                    ]
                    if missing_vars:
                        ds_batch.close()
                        raise ValueError(
                            f"Batch {os.path.basename(batch_file)} missing variables: "
                            f"{missing_vars}"
                        )
                    ds_subset = ds_batch[target_vars_to_fetch]
                    ds_selected = ds_subset.sel(
                        basin=common_basins, time=slice(t_range[0], t_range[1])
                    )
                    relevant_datasets.append(ds_selected)
                ds_batch.close()
            except Exception as e:
                print(f"Failed to read batch file {batch_file}: {e}")
                continue

        if not relevant_datasets:
            raise ValueError(
                f"Specified stations not found in any batch files: {gage_id_lst}"
            )

        if len(relevant_datasets) == 1:
            final_ds = relevant_datasets[0]
        else:
            final_ds = xr.concat(relevant_datasets, dim="basin")

        final_ds = final_ds.rename(rename_map)
        existing_basins = [b for b in gage_id_lst if b in final_ds.basin.values]
        if existing_basins:
            final_ds = final_ds.sel(basin=existing_basins)
        return final_ds
