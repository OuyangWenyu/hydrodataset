# -*- coding: utf-8 -*-
"""Local vs cloud read-speed comparison (run on ECS for cloud mode).

Usage:
    uv run python scripts/bench_cloud.py local     # PC: read local nc cache
    uv run python scripts/bench_cloud.py cloud     # ECS: read OSS nc cache
    uv run python scripts/bench_cloud.py zarr      # ECS: read OSS zarr cache
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

# ── Config ────────────────────────────────────────────────────────────

STATION = "01013500"
VAR_STATIC = "p_mean"
T_RANGE = ["1990-01-01", "1990-12-31"]
N_RUNS = 3

# Cache and OSS settings come from ~/hydro_setting.yml. Environment variables
# remain optional overrides for compatibility with the original benchmark.
try:
    from hydrodataset.configs.settings import get_cache_dir, get_s3_config

    LOCAL_CACHE = Path(get_cache_dir())
    _S3_SETTINGS = get_s3_config()
except Exception:
    LOCAL_CACHE = Path(os.getenv("LOCAL_CACHE_DIR", "D:/netcdf"))
    _S3_SETTINGS = {}


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


def _default_cloud_cache():
    bucket = str(_S3_SETTINGS.get("bucket", "")).strip("/")
    prefix = str(_S3_SETTINGS.get("prefix", "")).strip("/")
    if not bucket:
        return "camels-us/cache"
    return "/".join(part for part in (bucket, prefix, "zarr") if part)


CLOUD_CACHE = os.getenv("CLOUD_CACHE_PATH", _default_cloud_cache())


# ── Helpers ────────────────────────────────────────────────────────────


def get_s3fs():
    import s3fs

    return s3fs.S3FileSystem(**S3_CONFIG)


def open_ds(path, fmt="nc"):
    """Open a dataset — nc or zarr, local or s3://."""
    if fmt == "zarr":
        if str(path).startswith("s3://"):
            return xr.open_dataset(
                path,
                engine="zarr",
                storage_options=S3_CONFIG,
                consolidated=False,
                chunks={},
            )
        return xr.open_dataset(path, engine="zarr", consolidated=False, chunks={})
    if str(path).startswith("s3://"):
        from io import BytesIO
        fs = get_s3fs()
        return xr.open_dataset(BytesIO(fs.cat_file(path)), engine="h5netcdf")
    return xr.open_dataset(path)


def timed(label, fn, n=N_RUNS):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
    avg = np.mean(times)
    print(f"  {label}: {avg:.3f}s (avg over {n} runs)")
    if len(times) > 1:
        print(f"    min={min(times):.3f}s  max={max(times):.3f}s")
    return result


# ── Main ───────────────────────────────────────────────────────────────


def main(mode: str):
    if mode == "local":
        cache = LOCAL_CACHE
        attr_path = cache / "camels_us_attributes.nc"
        ts_path = cache / "camels_us_timeseries.nc"
    elif mode == "zarr":
        cache = CLOUD_CACHE
        attr_path = f"s3://{cache}/camels_us_attributes.zarr"
        ts_path = f"s3://{cache}/camels_us_timeseries.zarr"
    else:
        cache = CLOUD_CACHE
        attr_path = f"s3://{cache}/camels_us_attributes.nc"
        ts_path = f"s3://{cache}/camels_us_timeseries.nc"

    print(f"Mode: {mode}")
    print(f"Attributes: {attr_path}")
    print(f"Timeseries:  {ts_path}")
    print(f"Station:    {STATION}")
    print()

    # ── Static attributes ──
    print("═══ Static attributes ═══")

    fmt = "zarr" if mode == "zarr" else "nc"
    attr_ds = timed(f"open attrs {fmt}", lambda: open_ds(attr_path, fmt))
    print(f"    dims: {dict(attr_ds.sizes)}")

    def read_p_mean():
        return float(attr_ds[VAR_STATIC].sel(basin=STATION).values)

    v = timed(f"  read {VAR_STATIC} for {STATION}", read_p_mean)
    print(f"    → {VAR_STATIC} = {v:.4f}")

    # ── Dynamic timeseries ──
    print()
    print("═══ Dynamic timeseries ═══")

    ts_ds = timed(f"open ts {fmt}", lambda: open_ds(ts_path, fmt))
    print(f"    dims: {dict(ts_ds.sizes)}")

    # cloud cache uses raw var name q_cms_obs, local uses streamflow
    sf_key = "streamflow" if "streamflow" in ts_ds else "q_cms_obs"

    def read_streamflow():
        vals = ts_ds[sf_key].sel(
            basin=STATION, time=slice(T_RANGE[0], T_RANGE[1])
        )
        return float(vals.mean())

    v = timed(f"  read {sf_key} ({T_RANGE[0]}~{T_RANGE[1]})", read_streamflow)
    print(f"    → mean = {v:.4f}")

    attr_ds.close()
    ts_ds.close()

    print()
    print("Done.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "local"
    if mode not in ("local", "cloud", "zarr"):
        print("Usage: uv run python scripts/bench_cloud.py [local|cloud|zarr]")
        sys.exit(1)
    main(mode)
