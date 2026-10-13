# -*- coding: utf-8 -*-
"""Reconstruct the CAMELSH benchmark recorded in cloud_bench_result.md.

The four modes correspond to the columns in the historical table:

* local-batch: local attributes NC + batch001 timeseries NC
* local-api: local attributes NC + Camelsh.read_ts_xrdataset (scans batches)
* cloud-nc: attributes NC + batch001 NC downloaded in full from OSS
* cloud-zarr: attributes and timeseries read on demand from OSS Zarr

Defaults reproduce the recorded query: station 01013500, p_mean, and one year
of streamflow (1990). Each operation is averaged over three runs.
"""

from __future__ import annotations

import argparse
import os
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import xarray as xr

from hydrodataset import resolve_data_path
from hydrodataset.camelsh import Camelsh
from hydrodataset.configs.settings import get_cache_dir, get_s3_config


STATION = "01013500"
VAR_STATIC = "p_mean"
T_RANGE = ["1990-01-01", "1990-12-31"]
N_RUNS = 3

LOCAL_CACHE = Path(get_cache_dir())
_S3_SETTINGS = get_s3_config()


def _config_value(env_name, setting_name, default=None):
    env_value = os.getenv(env_name)
    if env_value is not None:
        return env_value
    return _S3_SETTINGS.get(setting_name, default)


S3_CONFIG = dict(
    key=_config_value("OSS_ACCESS_KEY_ID", "access_key_id"),
    secret=_config_value("OSS_ACCESS_KEY_SECRET", "secret_access_key"),
    client_kwargs={"region_name": os.getenv("OSS_REGION", "cn-beijing")},
    config_kwargs={"s3": {"addressing_style": "virtual"}},
    endpoint_url=_config_value("OSS_ENDPOINT", "endpoint_url"),
)


def _normalise_s3_root(value: str) -> str:
    return value.removeprefix("s3://").strip("/")


def _default_zarr_root() -> str:
    bucket = str(_S3_SETTINGS.get("bucket", "")).strip("/")
    prefix = str(_S3_SETTINGS.get("prefix", "")).strip("/")
    if not bucket:
        return "camelsh/cache"
    return "/".join(part for part in (bucket, prefix, "zarr") if part)


LOCAL_ATTR_PATH = LOCAL_CACHE / "camelsh_attributes.nc"
LOCAL_BATCH_PATH = LOCAL_CACHE / os.getenv(
    "CAMELSH_LOCAL_BATCH_FILE", "batch001_camelsh_timeseries.nc"
)
CLOUD_NC_ROOT = _normalise_s3_root(
    os.getenv("CAMELSH_CLOUD_NC_PATH", "camelsh/cache")
)
CLOUD_ZARR_ROOT = _normalise_s3_root(
    os.getenv("CAMELSH_ZARR_PATH", _default_zarr_root())
)


def _require_s3_settings() -> None:
    missing = [
        name
        for name, value in (
            ("access_key_id", S3_CONFIG["key"]),
            ("secret_access_key", S3_CONFIG["secret"]),
            ("endpoint_url", S3_CONFIG["endpoint_url"]),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing storage.s3 settings in ~/hydro_setting.yml: "
            + ", ".join(missing)
        )


def get_s3fs():
    import s3fs

    _require_s3_settings()
    return s3fs.S3FileSystem(**S3_CONFIG)


def open_ds(path, fmt="nc"):
    """Open local NC, whole-object OSS NC, or chunked OSS Zarr."""
    path_str = str(path)
    if fmt == "zarr":
        _require_s3_settings()
        return xr.open_dataset(
            path_str,
            engine="zarr",
            storage_options=S3_CONFIG,
            consolidated=False,
            chunks={},
        )
    if path_str.startswith("s3://"):
        content = get_s3fs().cat_file(path_str)
        return xr.open_dataset(BytesIO(content), engine="h5netcdf")
    return xr.open_dataset(path)


def timed(label, fn, n=N_RUNS):
    """Run an operation repeatedly and return its final result and mean time."""
    times = []
    previous = None
    for _ in range(n):
        if previous is not None and hasattr(previous, "close"):
            previous.close()
        t0 = time.perf_counter()
        previous = fn()
        times.append(time.perf_counter() - t0)
    avg = float(np.mean(times))
    print(f"  {label}: {avg:.3f}s (avg over {n} runs)")
    print(f"    min={min(times):.3f}s  max={max(times):.3f}s")
    return previous, avg


def _select_basin(data_array, station):
    basin_values = [str(value) for value in data_array["basin"].values]
    try:
        basin_index = basin_values.index(str(station))
    except ValueError as exc:
        raise KeyError(f"Station {station} is not present in the dataset") from exc
    return data_array.isel(basin=basin_index)


def _streamflow_name(dataset) -> str:
    for name in ("streamflow", "q_cms_obs"):
        if name in dataset.data_vars:
            return name
    raise KeyError("Neither streamflow nor q_cms_obs exists in the dataset")


def _direct_paths(mode):
    if mode == "local-batch":
        return LOCAL_ATTR_PATH, LOCAL_BATCH_PATH, "nc"
    if mode == "cloud-nc":
        return (
            f"s3://{CLOUD_NC_ROOT}/camelsh_attributes.nc",
            f"s3://{CLOUD_NC_ROOT}/batch001_camelsh_timeseries.nc",
            "nc",
        )
    if mode == "cloud-zarr":
        return (
            f"s3://{CLOUD_ZARR_ROOT}/camelsh_attributes.zarr",
            f"s3://{CLOUD_ZARR_ROOT}/camelsh_timeseries.zarr",
            "zarr",
        )
    raise ValueError(f"Mode {mode!r} does not use direct paths")


def _benchmark_attributes(path, fmt, station, repeats):
    print("\n═══ Static attributes ═══")
    attr_ds, open_time = timed(
        f"open attrs {fmt}", lambda: open_ds(path, fmt), repeats
    )
    print(f"    dims: {dict(attr_ds.sizes)}")

    def read_p_mean():
        return float(_select_basin(attr_ds[VAR_STATIC], station).values)

    value, read_time = timed(
        f"  read {VAR_STATIC} for {station}", read_p_mean, repeats
    )
    print(f"    → {VAR_STATIC} = {value:.4f}")
    return attr_ds, open_time, read_time


def _benchmark_timeseries(open_fn, label, station, t_range, repeats):
    print("\n═══ Dynamic timeseries ═══")
    ts_ds, open_time = timed(label, open_fn, repeats)
    print(f"    dims: {dict(ts_ds.sizes)}")
    streamflow_name = _streamflow_name(ts_ds)

    def read_streamflow():
        values = _select_basin(ts_ds[streamflow_name], station).sel(
            time=slice(t_range[0], t_range[1])
        )
        return float(values.mean())

    value, read_time = timed(
        f"  read {streamflow_name} ({t_range[0]}~{t_range[1]})",
        read_streamflow,
        repeats,
    )
    print(f"    → mean = {value:.4f}")
    return ts_ds, open_time, read_time


def benchmark(mode, station=STATION, t_range=T_RANGE, repeats=N_RUNS):
    print(f"Mode: {mode}")
    print(f"Station: {station}")

    if mode == "local-api":
        attr_path = LOCAL_ATTR_PATH
        fmt = "nc"
        uri = resolve_data_path("camelsh", source="local")
        reader = Camelsh(uri=uri)
        print(f"Attributes: {attr_path}")
        print("Timeseries: Camelsh.read_ts_xrdataset (scans local batch files)")

        def open_timeseries():
            return reader.read_ts_xrdataset(
                gage_id_lst=[station],
                t_range=t_range,
                var_lst=["streamflow"],
            )

        ts_label = "read_ts_xrdataset"
    else:
        attr_path, ts_path, fmt = _direct_paths(mode)
        print(f"Attributes: {attr_path}")
        print(f"Timeseries: {ts_path}")
        open_timeseries = lambda: open_ds(ts_path, fmt)
        ts_label = f"open ts {fmt}"

    attr_ds, attr_open, attr_read = _benchmark_attributes(
        attr_path, fmt, station, repeats
    )
    ts_ds, ts_open, ts_read = _benchmark_timeseries(
        open_timeseries, ts_label, station, t_range, repeats
    )

    attr_ds.close()
    ts_ds.close()
    total = attr_open + attr_read + ts_open + ts_read
    print(f"\nTotal: {total:.3f}s (sum of operation averages)")
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("local-batch", "local-api", "cloud-nc", "cloud-zarr"),
    )
    parser.add_argument("--station", default=STATION)
    parser.add_argument("--start", default=T_RANGE[0])
    parser.add_argument("--end", default=T_RANGE[1])
    parser.add_argument("--repeats", type=int, default=N_RUNS)
    args = parser.parse_args()
    benchmark(
        args.mode,
        station=args.station,
        t_range=[args.start, args.end],
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
