import os
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from hydrodataset import HydroDataset, StandardVariable


class CamelsPe(HydroDataset):
    """CAMELS-PE dataset reader.

    This reader currently supports local CAMELS-PE files. Cloud/zarr support can
    be added after the local CSV reader is validated.
    """

    _DATA_REL = "CAMELS_PE/CAMELS-PE"
    _ATTR_REL = f"{_DATA_REL}/02_attributes"
    _META_REL = f"{_DATA_REL}/01_metadata"
    _TS_REL = f"{_DATA_REL}/03_timeseries/by_catchment"

    _ATTR_FILES = [
        "climatic_indices.csv",
        "geologic_attributes.csv",
        "human_intervention_attributes.csv",
        "hydrological_signatures.csv",
        "landcover_attributes.csv",
        "soil_attributes.csv",
        "topographic_attributes.csv",
    ]

    _COL_MAP = {
        "prec": "pcp_mm",
        "prec_var": "pcp_var",
        "flow_obs": "q_mm_obs",
        "flow_sim": "q_mm_sim",
        "pet": "pet_mm",
        "tmin": "airtemp_c_min",
        "tmean": "airtemp_c_mean",
        "tmax": "airtemp_c_max",
        "srad": "solrad_mj_m2",
        "vprp": "vapor_pressure_hpa",
    }

    def __init__(
        self, uri: str, region: Optional[str] = None, download: bool = False
    ) -> None:
        super().__init__(uri)
        self.region = region
        self.download = download

    def get_name(self):
        return "CAMELS_PE"

    def set_data_source_describe(self):
        return {
            "metadata": os.path.join(
                str(self.data_source_dir), *self._META_REL.split("/")
            ),
            "attributes": os.path.join(
                str(self.data_source_dir), *self._ATTR_REL.split("/")
            ),
            "timeseries": os.path.join(
                str(self.data_source_dir), *self._TS_REL.split("/")
            ),
        }

    def download_data_source(self):
        raise NotImplementedError("CAMELS-PE download is not implemented yet.")

    def is_data_ready(self):
        root = self._dataset_root()
        return (
            os.path.isdir(root)
            and os.path.isfile(os.path.join(root, "01_metadata", "stations.csv"))
            and os.path.isdir(os.path.join(root, "02_attributes"))
            and os.path.isdir(os.path.join(root, "03_timeseries", "by_catchment"))
        )

    def read_object_ids(self) -> np.ndarray:
        if self._is_cloud():
            fs = self._make_s3fs()
            with fs.open(self._cloud_key(self._META_REL, "stations.csv")) as fh:
                stations = pd.read_csv(fh, dtype={"gauge_id": str})
        else:
            stations_path = os.path.join(self._metadata_dir(), "stations.csv")
            stations = pd.read_csv(stations_path, dtype={"gauge_id": str})
        return np.sort(stations["gauge_id"].astype(str).to_numpy())

    def get_constant_cols(self) -> np.ndarray:
        cols = []
        for fname in ["stations.csv", *self._ATTR_FILES]:
            if fname == "stations.csv":
                fpath = os.path.join(self._metadata_dir(), fname)
            else:
                fpath = os.path.join(self._attributes_dir(), fname)
            header = pd.read_csv(fpath, nrows=0).columns.tolist()
            cols.extend(c for c in header if c != "gauge_id")
        return np.array(self._clean_feature_names(cols))

    def get_relevant_cols(self) -> np.ndarray:
        return np.array(list(self._COL_MAP.values()))

    def get_target_cols(self) -> np.ndarray:
        return np.array(["q_mm_obs", "q_mm_sim"])

    def get_other_cols(self) -> dict:
        return {}

    def cache_attributes_xrdataset(self):
        static = self._read_static_frame()
        ds_attr = static.to_xarray()
        coord_name = list(ds_attr.sizes.keys())[0]
        if coord_name != "basin":
            ds_attr = ds_attr.rename({coord_name: "basin"})
        ds_attr = self._assign_units_to_dataset(ds_attr, self._get_attribute_units())
        ds_attr.to_netcdf(self.cache_dir.joinpath(self._attributes_cache_filename))

    def cache_attributes_to_zarr(self) -> None:
        import zarr

        static = self._read_static_frame()
        zarr_name = self._attributes_cache_filename.replace(".nc", ".zarr")
        out, opts = self._zarr_path_and_opts(zarr_name)
        n = len(static.index)
        root = zarr.open_group(out, mode="w", storage_options=opts, zarr_format=2)
        for col in static.columns:
            vals = (
                static[col].values.astype(str)
                if static[col].dtype == object
                else static[col].values
            )
            arr = root.create_array(col, shape=(n,), chunks=(n,), dtype=vals.dtype)
            arr[:] = vals
            arr.attrs["_ARRAY_DIMENSIONS"] = ["basin"]
        basin_arr = root.create_array("basin", shape=(n,), chunks=(n,), dtype=str)
        basin_arr[:] = static.index.astype(str).tolist()
        basin_arr.attrs["_ARRAY_DIMENSIONS"] = ["basin"]
        root.attrs["coordinates"] = "basin"
        self._write_zarr_units(root, "static")
        print(f"Attributes zarr written to: {out}")

    def cache_timeseries_xrdataset(self):
        stations = self.read_object_ids().tolist()
        all_times = pd.date_range(
            self.default_t_range[0], self.default_t_range[1], freq="D"
        )
        n, nt = len(stations), len(all_times)
        data = {
            clean_name: np.full((n, nt), np.nan, dtype="float64")
            for clean_name in self._COL_MAP.values()
        }

        ts_dir = self._timeseries_dir()
        for i, station in enumerate(tqdm(stations, desc="CAMELS_PE cache")):
            fpath = os.path.join(ts_dir, f"{station}.csv")
            if not os.path.isfile(fpath):
                continue
            df = pd.read_csv(
                fpath, index_col="date", parse_dates=True, na_values=["NA"]
            )
            df = df.reindex(all_times)
            for raw_col, clean_name in self._COL_MAP.items():
                if raw_col in df.columns:
                    data[clean_name][i] = pd.to_numeric(
                        df[raw_col], errors="coerce"
                    ).to_numpy(dtype="float64")

        ds = xr.Dataset(
            data_vars={
                name: (("basin", "time"), values) for name, values in data.items()
            },
            coords={"basin": stations, "time": all_times},
        )
        units = {}
        for info in self._dynamic_variable_mapping.values():
            for source in info["sources"].values():
                units[source["specific_name"]] = source["unit"]
        for name in ds.data_vars:
            ds[name].attrs["units"] = units.get(name, "unknown")
        ds.to_netcdf(self.cache_dir.joinpath(self._timeseries_cache_filename))

    def cache_timeseries_to_zarr(self) -> None:
        import zarr

        fs = self._make_s3fs()
        stations = self.read_object_ids().tolist()
        all_times = pd.date_range(
            self.default_t_range[0], self.default_t_range[1], freq="D"
        )
        n, nt = len(stations), len(all_times)
        times_ns = all_times.asi8
        all_vars = list(self._COL_MAP.values())
        data = {vn: np.full((n, nt), np.nan, dtype="float64") for vn in all_vars}

        print(f"Reading {n} CAMELS-PE stations from OSS...")
        for i, station in enumerate(tqdm(stations, desc="CAMELS_PE zarr")):
            path = self._cloud_key(self._TS_REL, f"{station}.csv")
            try:
                with fs.open(path) as fh:
                    df = pd.read_csv(
                        fh, index_col="date", parse_dates=True, na_values=["NA"]
                    )
                df = df.reindex(all_times)
                for raw_col, clean_name in self._COL_MAP.items():
                    if raw_col in df.columns:
                        data[clean_name][i] = pd.to_numeric(
                            df[raw_col], errors="coerce"
                        ).to_numpy(dtype="float64")
            except Exception as e:
                print(f"  WARN {station}: {e}")

        zarr_name = self._timeseries_cache_filename.replace(".nc", ".zarr")
        out, opts = self._zarr_path_and_opts(zarr_name)
        root = zarr.open_group(out, mode="w", storage_options=opts, zarr_format=2)
        for vn in all_vars:
            arr = root.create_array(
                vn,
                shape=(n, nt),
                chunks=(min(n, 100), min(nt, 365)),
                dtype="float64",
            )
            arr[:] = data[vn]
            arr.attrs["_ARRAY_DIMENSIONS"] = ["basin", "time"]

        time_arr = root.create_array(
            "time",
            shape=(nt,),
            chunks=(min(nt, 365),),
            dtype="int64",
        )
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

    def read_target_cols(
        self,
        object_ids=None,
        t_range_list=None,
        target_cols=None,
        gage_id_lst=None,
        t_range=None,
        **kwargs,
    ) -> np.ndarray:
        if object_ids is None:
            object_ids = gage_id_lst
        if t_range_list is None:
            t_range_list = t_range
        if target_cols is None:
            target_cols = ["streamflow"]
        ds = self.read_ts_xrdataset(
            gage_id_lst=object_ids,
            t_range=t_range_list,
            var_lst=target_cols,
            **kwargs,
        )
        return ds.to_array().transpose("basin", "time", "variable").values

    def read_relevant_cols(
        self,
        object_ids=None,
        t_range_list=None,
        relevant_cols=None,
        gage_id_lst=None,
        t_range=None,
        var_lst=None,
        forcing_type=None,
        **kwargs,
    ) -> np.ndarray:
        if object_ids is None:
            object_ids = gage_id_lst
        if t_range_list is None:
            t_range_list = t_range
        if relevant_cols is None:
            relevant_cols = var_lst
        ds = self.read_ts_xrdataset(
            gage_id_lst=object_ids,
            t_range=t_range_list,
            var_lst=relevant_cols,
            **kwargs,
        )
        return ds.to_array().transpose("basin", "time", "variable").values

    def read_constant_cols(
        self,
        object_ids=None,
        constant_cols=None,
        gage_id_lst=None,
        var_lst=None,
        **kwargs,
    ) -> np.ndarray:
        if object_ids is None:
            object_ids = gage_id_lst
        if constant_cols is None:
            constant_cols = var_lst
        ds = self.read_attr_xrdataset(
            gage_id_lst=object_ids,
            var_lst=constant_cols,
            **kwargs,
        )
        return ds.to_array().transpose("basin", "variable").values

    def read_other_cols(
        self, object_ids=None, other_cols: dict = None, **kwargs
    ) -> dict:
        return {}

    @property
    def _attributes_cache_filename(self):
        return "camels_pe_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camels_pe_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1981-01-01", "2025-12-31"]

    def _read_static_frame(self) -> pd.DataFrame:
        frames = []
        if self._is_cloud():
            fs = self._make_s3fs()
            with fs.open(self._cloud_key(self._META_REL, "stations.csv")) as fh:
                stations = pd.read_csv(fh, dtype={"gauge_id": str}).set_index(
                    "gauge_id"
                )
            frames.append(stations)
            for fname in self._ATTR_FILES:
                with fs.open(self._cloud_key(self._ATTR_REL, fname)) as fh:
                    df = pd.read_csv(fh, dtype={"gauge_id": str}).set_index("gauge_id")
                frames.append(df)
        else:
            stations = pd.read_csv(
                os.path.join(self._metadata_dir(), "stations.csv"),
                dtype={"gauge_id": str},
            ).set_index("gauge_id")
            frames.append(stations)
            for fname in self._ATTR_FILES:
                df = pd.read_csv(
                    os.path.join(self._attributes_dir(), fname),
                    dtype={"gauge_id": str},
                ).set_index("gauge_id")
                frames.append(df)

        static = pd.concat(frames, axis=1)
        static = static.loc[:, ~static.columns.duplicated()]
        static.index = static.index.astype(str)
        static = static.reindex(self.read_object_ids().tolist())
        static.columns = self._clean_feature_names(static.columns)
        return static.rename(
            columns={
                "area": "area_km2",
                "gauge_lat": "lat",
                "gauge_lon": "long",
            }
        )

    def _cloud_key(self, *parts):
        uri = str(self.data_source_dir).rstrip("/")
        rel = "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))
        return f"{uri}/{rel}".removeprefix("s3://")

    def _dataset_root(self):
        return os.path.join(str(self.data_source_dir), *self._DATA_REL.split("/"))

    def _metadata_dir(self):
        return os.path.join(str(self.data_source_dir), *self._META_REL.split("/"))

    def _attributes_dir(self):
        return os.path.join(str(self.data_source_dir), *self._ATTR_REL.split("/"))

    def _timeseries_dir(self):
        return os.path.join(str(self.data_source_dir), *self._TS_REL.split("/"))

    _subclass_static_definitions = {
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
        "gauge_lat": {"specific_name": "lat", "unit": "degree"},
        "gauge_lon": {"specific_name": "long", "unit": "degree"},
        "gauge_elev": {"specific_name": "gauge_elev", "unit": "m"},
        "elev_mean": {"specific_name": "elev_mean", "unit": "m"},
        "pet_mean": {"specific_name": "pet_mean", "unit": "mm/day"},
        "q_mean": {"specific_name": "q_mean", "unit": "mm/day"},
    }

    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "q_mm_obs", "unit": "mm/day"},
                "simulation": {"specific_name": "q_mm_sim", "unit": "mm/day"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "pisco",
            "sources": {
                "pisco": {"specific_name": "pcp_mm", "unit": "mm/day"},
            },
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "pisco",
            "sources": {
                "pisco": {"specific_name": "pet_mm", "unit": "mm/day"},
            },
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "pisco",
            "sources": {
                "pisco": {"specific_name": "airtemp_c_min", "unit": "degC"},
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "pisco",
            "sources": {
                "pisco": {"specific_name": "airtemp_c_mean", "unit": "degC"},
            },
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "pisco",
            "sources": {
                "pisco": {"specific_name": "airtemp_c_max", "unit": "degC"},
            },
        },
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "era5_land",
            "sources": {
                "era5_land": {"specific_name": "solrad_mj_m2", "unit": "MJ/m^2/day"},
            },
        },
        StandardVariable.VAPOR_PRESSURE: {
            "default_source": "era5_land",
            "sources": {
                "era5_land": {"specific_name": "vapor_pressure_hpa", "unit": "hPa"},
            },
        },
    }
