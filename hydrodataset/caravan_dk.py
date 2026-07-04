import os
from typing import Optional

import numpy as np
import pandas as pd

from aqua_fetch import Caravan_DK
from hydrodataset import HydroDataset, StandardVariable


class CaravanDK(HydroDataset):
    """Caravan_DK dataset class extending HydroDataset.

    This class uses a custom data reading implementation to support a newer
    dataset version than the one supported by the underlying aquafetch library.
    It overrides the download URLs and provides its own parsing and caching logic.
    """

    def __init__(
        self, uri: str, region: Optional[str] = None, download: bool = False
    ) -> None:
        """Initialize Caravan_DK dataset.

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

        # Define the new URLs for the latest dataset version
        new_url = "https://zenodo.org/records/15200118"

        def do_nothing(self, *args, **kwargs):
            pass

        def custom_boundary_file(self) -> os.PathLike:
            return os.path.join(
                self.path, "shapefiles", "camelsdk", "camelsdk_basin_shapes.shp"
            )

        def custom_csv_path(self):
            return os.path.join(self.path, "timeseries", "csv", "camelsdk")

        def custom_nc_path(self):
            return os.path.join(self.path, "timeseries", "netcdf", "camelsdk")

        def custom_other_attr_fpath(self):
            """returns path to attributes_other_camelsdk.csv file"""
            return os.path.join(
                self.path, "attributes", "camelsdk", "attributes_other_camelsdk.csv"
            )

        def custom_caravan_attr_fpath(self):
            """returns path to attributes_caravan_camelsdk.csv file"""
            return os.path.join(
                self.path, "attributes", "camelsdk", "attributes_caravan_camelsdk.csv"
            )

        def custom_hyd_atlas_fpath(self):
            return os.path.join(
                self.path,
                "attributes",
                "camelsdk",
                "attributes_hydroatlas_camelsdk.csv",
            )

        # Create class attributes dictionary for dynamic class creation
        class_attrs = {
            "url": new_url,
            "boundary_file": property(custom_boundary_file),
            "csv_path": property(custom_csv_path),
            "nc_path": property(custom_nc_path),
            "other_attr_fpath": property(custom_other_attr_fpath),
            "caravan_attr_fpath": property(custom_caravan_attr_fpath),
            "hyd_atlas_fpath": property(custom_hyd_atlas_fpath),
            "_maybe_to_netcdf": do_nothing,
        }

        # Create a custom Caravan_DK class using type() to preserve the class name
        CustomCaravanDK = type("Caravan_DK", (Caravan_DK,), class_attrs)

        # Instantiate our custom class
        self.aqua_fetch = CustomCaravanDK(uri)

    # OSS relative paths (uploaded raw data folder: Caravan_DK)
    _ATTR_REL = "Caravan_DK/attributes/camelsdk"
    _CSV_REL = "Caravan_DK/timeseries/csv/camelsdk"
    # AquaFetch Caravan_DK.static_map
    _STATIC_RENAME = {
        "area": "area_km2",
        "gauge_lat": "lat",
        "slope_mean": "slope_mkm-1",
        "gauge_lon": "long",
    }

    def read_object_ids(self) -> np.ndarray:
        if self._is_cloud():
            fs = self._make_s3fs()
            uri = str(self.data_source_dir).rstrip("/")
            names = [p.split("/")[-1] for p in fs.ls(f"{uri}/{self._CSV_REL}".removeprefix("s3://"))]
            ids = sorted(
                n.split(".csv")[0][9:] for n in names if n.startswith("camelsdk_")
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
                df = pd.read_csv(fh)
            gid = df.pop("gauge_id")
            df.index = [str(i)[9:] for i in gid]
            return df

        hyd = _read("attributes_hydroatlas_camelsdk.csv")
        other = _read("attributes_other_camelsdk.csv")
        caravan = _read("attributes_caravan_camelsdk.csv")
        static = pd.concat([hyd, other, caravan], axis=1)
        static = static.loc[~static.index.duplicated(keep="first")]
        static = static.rename(columns=self._STATIC_RENAME)
        static.index = static.index.astype(str)
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

    def cache_timeseries_to_zarr(self, batch_size: int = 60) -> None:
        import zarr
        from tqdm import tqdm

        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")
        ts_base = f"{uri}/{self._CSV_REL}"

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
                path = f"{ts_base}/camelsdk_{stn}.csv".removeprefix("s3://")
                try:
                    with fs.open(path) as fh:
                        df = pd.read_csv(fh)
                    df.index = pd.to_datetime(df.pop("date"))
                    df = df.rename(columns={"streamflow": "q_cms_obs"})
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
        self._write_zarr_units(root, "dynamic")
        print(f"Timeseries zarr written to: {out}")

    @property
    def _attributes_cache_filename(self):
        return "caravan_dk_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "caravan_dk_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1981-01-02", "2020-12-31"]

    # Define standardized static variable mappings
    # These variables are already present in the dataset, so we just map them
    # get the information of features from "https://essd.copernicus.org/articles/17/1551/2025/essd-17-1551-2025.pdf"
    _subclass_static_definitions = {
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
    }

    # Define standardized dynamic variable mappings
    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "q_cms_obs", "unit": "m^3/s"}
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "era5_land",
            "sources": {
                "era5_land": {
                    "specific_name": "total_precipitation_sum",
                    "unit": "mm/day",
                }
            },
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "era5_land",
            "sources": {
                "era5_land": {"specific_name": "temperature_2m_max", "unit": "°C"},
                "dewpoint": {
                    "specific_name": "dewpoint_temperature_2m_max",
                    "unit": "°C",
                },
            },
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "era5_land",
            "sources": {
                "era5_land": {"specific_name": "temperature_2m_min", "unit": "°C"},
                "dewpoint": {
                    "specific_name": "dewpoint_temperature_2m_min",
                    "unit": "°C",
                },
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "era5_land",
            "sources": {
                "era5_land": {"specific_name": "temperature_2m_mean", "unit": "°C"},
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
                "fao_penman_monteith": {
                    "specific_name": "potential_evaporation_sum_fao_penman_monteith",
                    "unit": "mm/day",
                },
            },
        },
        StandardVariable.SNOW_WATER_EQUIVALENT: {
            "default_source": "mean",
            "sources": {
                "mean": {
                    "specific_name": "snow_depth_water_equivalent_mean",
                    "unit": "mm",
                },
            },
        },
        StandardVariable.SNOW_WATER_EQUIVALENT_MIN: {
            "default_source": "min",
            "sources": {
                "min": {
                    "specific_name": "snow_depth_water_equivalent_min",
                    "unit": "mm",
                },
            },
        },
        StandardVariable.SNOW_WATER_EQUIVALENT_MAX: {
            "default_source": "max",
            "sources": {
                "max": {
                    "specific_name": "snow_depth_water_equivalent_max",
                    "unit": "mm",
                },
            },
        },
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "mean",
            "sources": {
                "mean": {
                    "specific_name": "surface_net_solar_radiation_mean",
                    "unit": "W/m^2",
                },
            },
        },
        StandardVariable.SOLAR_RADIATION_MIN: {
            "default_source": "min",
            "sources": {
                "min": {
                    "specific_name": "surface_net_solar_radiation_min",
                    "unit": "W/m^2",
                },
            },
        },
        StandardVariable.SOLAR_RADIATION_MAX: {
            "default_source": "max",
            "sources": {
                "max": {
                    "specific_name": "surface_net_solar_radiation_max",
                    "unit": "W/m^2",
                },
            },
        },
        StandardVariable.THERMAL_RADIATION_MIN: {
            "default_source": "min",
            "sources": {
                "min": {
                    "specific_name": "surface_net_thermal_radiation_min",
                    "unit": "W/m^2",
                },
            },
        },
        StandardVariable.THERMAL_RADIATION_MAX: {
            "default_source": "max",
            "sources": {
                "max": {
                    "specific_name": "surface_net_thermal_radiation_max",
                    "unit": "W/m^2",
                },
            },
        },
        StandardVariable.THERMAL_RADIATION: {
            "default_source": "mean",
            "sources": {
                "mean": {
                    "specific_name": "surface_net_thermal_radiation_mean",
                    "unit": "W/m^2",
                },
            },
        },
        StandardVariable.SURFACE_PRESSURE_MIN: {
            "default_source": "min",
            "sources": {
                "min": {"specific_name": "surface_pressure_min", "unit": "Pa"},
            },
        },
        StandardVariable.SURFACE_PRESSURE_MAX: {
            "default_source": "max",
            "sources": {
                "max": {"specific_name": "surface_pressure_max", "unit": "Pa"},
            },
        },
        StandardVariable.SURFACE_PRESSURE: {
            "default_source": "mean",
            "sources": {
                "mean": {"specific_name": "surface_pressure_mean", "unit": "Pa"},
            },
        },
        StandardVariable.U_WIND_SPEED_MIN: {
            "default_source": "min",
            "sources": {
                "min": {"specific_name": "u_component_of_wind_10m_min", "unit": "m/s"},
            },
        },
        StandardVariable.U_WIND_SPEED_MAX: {
            "default_source": "max",
            "sources": {
                "max": {"specific_name": "u_component_of_wind_10m_max", "unit": "m/s"},
            },
        },
        StandardVariable.U_WIND_SPEED: {
            "default_source": "mean",
            "sources": {
                "mean": {
                    "specific_name": "u_component_of_wind_10m_mean",
                    "unit": "m/s",
                },
            },
        },
        StandardVariable.V_WIND_SPEED_MIN: {
            "default_source": "min",
            "sources": {
                "min": {"specific_name": "v_component_of_wind_10m_min", "unit": "m/s"},
            },
        },
        StandardVariable.V_WIND_SPEED_MAX: {
            "default_source": "max",
            "sources": {
                "max": {"specific_name": "v_component_of_wind_10m_max", "unit": "m/s"},
            },
        },
        StandardVariable.V_WIND_SPEED: {
            "default_source": "mean",
            "sources": {
                "mean": {
                    "specific_name": "v_component_of_wind_10m_mean",
                    "unit": "m/s",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER1_MIN: {
            "default_source": "min",
            "sources": {
                "min": {
                    "specific_name": "volumetric_soil_water_layer_1_min",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER1_MAX: {
            "default_source": "max",
            "sources": {
                "max": {
                    "specific_name": "volumetric_soil_water_layer_1_max",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER1: {
            "default_source": "mean",
            "sources": {
                "mean": {
                    "specific_name": "volumetric_soil_water_layer_1_mean",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER2_MIN: {
            "default_source": "min",
            "sources": {
                "min": {
                    "specific_name": "volumetric_soil_water_layer_2_min",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER2_MAX: {
            "default_source": "max",
            "sources": {
                "max": {
                    "specific_name": "volumetric_soil_water_layer_2_max",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER2: {
            "default_source": "mean",
            "sources": {
                "mean": {
                    "specific_name": "volumetric_soil_water_layer_2_mean",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER3_MIN: {
            "default_source": "min",
            "sources": {
                "min": {
                    "specific_name": "volumetric_soil_water_layer_3_min",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER3_MAX: {
            "default_source": "max",
            "sources": {
                "max": {
                    "specific_name": "volumetric_soil_water_layer_3_max",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER3: {
            "default_source": "mean",
            "sources": {
                "mean": {
                    "specific_name": "volumetric_soil_water_layer_3_mean",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER4_MIN: {
            "default_source": "min",
            "sources": {
                "min": {
                    "specific_name": "volumetric_soil_water_layer_4_min",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER4_MAX: {
            "default_source": "max",
            "sources": {
                "max": {
                    "specific_name": "volumetric_soil_water_layer_4_max",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER4: {
            "default_source": "mean",
            "sources": {
                "mean": {
                    "specific_name": "volumetric_soil_water_layer_4_mean",
                    "unit": "m^3/m^3",
                },
            },
        },
    }
