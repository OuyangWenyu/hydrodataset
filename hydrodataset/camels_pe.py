import os
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from aqua_fetch import CAMELS_PE
from hydrodataset import HydroDataset, StandardVariable


class CamelsPe(HydroDataset):
    """CAMELS-PE dataset reader.

    Thin wrapper over the aqua_fetch ``CAMELS_PE`` class (available since
    aqua-fetch 1.1.0). Data is downloaded/extracted by aqua_fetch to
    ``{root}/CAMELS_PE/CAMELS-PE_v1.0.1/CAMELS-PE/...`` and cached locally as
    ``camels_pe_attributes.nc`` / ``camels_pe_timeseries.nc`` via the base
    ``HydroDataset`` cache methods.

    Cloud (S3/OSS) support: the base ``HydroDataset`` reads a cloud zarr store
    at ``s3://bucket/zarr/camels_pe_*.zarr`` and, when it is missing, calls
    ``cache_*_to_zarr`` to auto-generate it. ``CamelsPe`` implements these by
    reading the raw CAMELS-PE files directly from OSS (aqua_fetch cannot read
    S3), mirroring the other ``camels_*`` readers.
    """

    def __init__(
        self, uri: str, region: Optional[str] = None, download: bool = False
    ) -> None:
        super().__init__(uri)
        self.region = region
        self.download = download
        if not str(uri).startswith("s3://"):
            self.aqua_fetch = CAMELS_PE(uri)

    def get_name(self):
        return "CAMELS_PE"

    def set_data_source_describe(self):
        root = os.path.join(
            str(self.data_source_dir), "CAMELS_PE", "CAMELS-PE_v1.0.1", "CAMELS-PE"
        )
        return {
            "metadata": os.path.join(root, "01_metadata"),
            "attributes": os.path.join(root, "02_attributes"),
            "timeseries": os.path.join(root, "03_timeseries"),
        }

    def download_data_source(self):
        if not hasattr(self, "aqua_fetch"):
            raise NotImplementedError("cloud download is not supported; use cache_*_to_zarr")
        self.aqua_fetch._download_camels_pe(overwrite=True)

    def read_object_ids(self) -> np.ndarray:
        """Read station IDs.

        Local: delegate to the aqua_fetch wrapper's ``stations()``. Cloud (S3):
        read ``stations.csv`` from the metadata dir directly (no aqua_fetch).
        """
        if not self._is_cloud():
            if hasattr(self, "aqua_fetch"):
                return np.sort(np.array(self.aqua_fetch.stations()))
            raise NotImplementedError
        fs = self._make_s3fs()
        rel = "CAMELS_PE/CAMELS-PE_v1.0.1/CAMELS-PE/01_metadata/stations.csv"
        uri = str(self.data_source_dir).rstrip("/")
        with fs.open(f"{uri}/{rel}".removeprefix("s3://")) as fh:
            stations = pd.read_csv(fh, dtype={"gauge_id": str})
        return np.sort(stations["gauge_id"].astype(str).to_numpy())

    def _oss_rel_base(self) -> str:
        """OSS-relative root of the extracted CAMELS-PE dataset."""
        return "CAMELS_PE/CAMELS-PE_v1.0.1/CAMELS-PE"

    def cache_attributes_to_zarr(self) -> None:
        """Generate ``camels_pe_attributes.zarr`` on OSS from the raw attribute
        files (no aqua_fetch, which cannot read S3). Mirrors the local
        attribute cache: standard columns only, basin coordinate included.
        """
        import zarr

        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")
        base = self._oss_rel_base()

        def read_csv(rel: str, **kw) -> pd.DataFrame:
            with fs.open(f"{uri}/{rel}".removeprefix("s3://")) as fh:
                return pd.read_csv(fh, **kw)

        frames = [
            read_csv(f"{base}/01_metadata/stations.csv", index_col="gauge_id", dtype={"gauge_id": str})
        ]
        for name in (
            "topographic_attributes",
            "climatic_indices",
            "hydrological_signatures",
            "landcover_attributes",
            "geologic_attributes",
            "soil_attributes",
            "human_intervention_attributes",
        ):
            frames.append(
                read_csv(f"{base}/02_attributes/{name}.csv", index_col="gauge_id", dtype={"gauge_id": str})
            )

        df = pd.concat(frames, axis=1)
        df = df.rename(
            columns={
                "area": "area_km2",
                "elev_mean": "elev_catch_m",
                "gauge_lat": "lat",
                "gauge_lon": "long",
                "gauge_elev": "elev_gauge_m",
            }
        )
        df.index = df.index.astype(str)
        df.index.name = "basin"

        std_cols = ["area_km2", "lat", "long", "elev_gauge_m", "elev_catch_m", "p_mean", "pet_mean", "q_mean"]
        df = df[[c for c in std_cols if c in df.columns]]
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        zarr_name = self._attributes_cache_filename.replace(".nc", ".zarr")
        out, opts = self._zarr_path_and_opts(zarr_name)
        root = zarr.open_group(out, mode="w", storage_options=opts, zarr_format=2)
        ids = df.index.tolist()
        n = len(ids)
        for col in df.columns:
            arr = root.create_array(col, shape=(n,), chunks=(n,), dtype="float64")
            arr[:] = df[col].values.astype(float)
            arr.attrs["_ARRAY_DIMENSIONS"] = ["basin"]
        basin_arr = root.create_array("basin", shape=(n,), chunks=(n,), dtype=str)
        basin_arr[:] = ids
        basin_arr.attrs["_ARRAY_DIMENSIONS"] = ["basin"]
        root.attrs["coordinates"] = "basin"
        self._write_zarr_units(root, "static")
        print(f"Attributes zarr written to: {out}")

    def cache_timeseries_to_zarr(self) -> None:
        """Generate ``camels_pe_timeseries.zarr`` on OSS from the raw
        per-catchment daily files (no aqua_fetch, which cannot read S3).
        Mirrors the local timeseries cache: standard variables, basin + time
        coordinates, times stored as nanoseconds since 1970-01-01.
        """
        import zarr

        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")
        base = self._oss_rel_base()

        rename = {
            "prec": "pcp_mm",
            "flow_obs": "q_mm_obs",
            "pet": "pet_mm",
            "tmin": "airtemp_c_min",
            "tmean": "airtemp_c_mean",
            "tmax": "airtemp_c_max",
            "vprp": "vp_hpa",
            "srad": "srad",
        }
        std_vars = ["q_mm_obs", "pcp_mm", "pet_mm", "airtemp_c_min", "airtemp_c_mean", "airtemp_c_max", "srad", "vp_hpa"]

        stations = self.read_object_ids().tolist()
        all_times = pd.date_range(self.default_t_range[0], self.default_t_range[1], freq="D")
        n, nt = len(stations), len(all_times)
        times_ns = all_times.asi8

        data = {vn: np.full((n, nt), np.nan) for vn in std_vars}
        for i, stn in enumerate(tqdm(stations, desc="CAMELS_PE zarr")):
            rel = f"{base}/03_timeseries/by_catchment/{stn}.csv"
            try:
                with fs.open(f"{uri}/{rel}".removeprefix("s3://")) as fh:
                    df = pd.read_csv(fh, index_col="date", parse_dates=True)
                df = df[~df.index.duplicated(keep="first")]
                df = df.rename(columns=rename)
                df = df.reindex(all_times)
                for vn in std_vars:
                    if vn in df.columns:
                        data[vn][i] = df[vn].values.astype(float)
            except Exception as e:
                print(f"  WARN {stn}: {e}")

        zarr_name = self._timeseries_cache_filename.replace(".nc", ".zarr")
        out, opts = self._zarr_path_and_opts(zarr_name)
        root = zarr.open_group(out, mode="w", storage_options=opts, zarr_format=2)

        for vn in std_vars:
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

    @property
    def _attributes_cache_filename(self):
        return "camels_pe_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camels_pe_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1981-01-01", "2025-12-31"]

    _subclass_static_definitions = {
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
        "gauge_lat": {"specific_name": "lat", "unit": "degree"},
        "gauge_lon": {"specific_name": "long", "unit": "degree"},
        "gauge_elev": {"specific_name": "elev_gauge_m", "unit": "m"},
        "elev_mean": {"specific_name": "elev_catch_m", "unit": "m"},
        "pet_mean": {"specific_name": "pet_mean", "unit": "mm/day"},
        "q_mean": {"specific_name": "q_mean", "unit": "mm/day"},
    }

    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "observations",
            # NOTE: aqua-fetch 1.1.0 exposes only observed streamflow
            # (q_mm_obs); model-simulated flow_sim is intentionally dropped.
            "sources": {
                "observations": {"specific_name": "q_mm_obs", "unit": "mm/day"},
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
                "era5_land": {"specific_name": "srad", "unit": "MJ/m^2/day"},
            },
        },
        StandardVariable.VAPOR_PRESSURE: {
            "default_source": "era5_land",
            "sources": {
                "era5_land": {"specific_name": "vp_hpa", "unit": "hPa"},
            },
        },
    }
