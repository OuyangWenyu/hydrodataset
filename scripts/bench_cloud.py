# -*- coding: utf-8 -*-
"""Local vs cloud read-speed comparison (run on ECS for cloud mode).

Usage:
    uv run python scripts/bench_cloud.py local     # PC: read local cache
    uv run python scripts/bench_cloud.py cloud     # ECS: read OSS cache
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import s3fs
import xarray as xr

# ── Config ────────────────────────────────────────────────────────────

STATION = "01013500"
VAR_STATIC = "p_mean"
T_RANGE = ["1990-01-01", "1990-12-31"]
N_RUNS = 3

# Cloud credentials — set via env vars, never hardcoded
S3_CONFIG = dict(
    key=os.environ["OSS_ACCESS_KEY_ID"],
    secret=os.environ["OSS_ACCESS_KEY_SECRET"],
    client_kwargs={"region_name": os.environ.get("OSS_REGION", "cn-beijing")},
    config_kwargs={"s3": {"addressing_style": "virtual"}},
    endpoint_url=os.environ.get(
        "OSS_ENDPOINT", "https://oss-cn-beijing-internal.aliyuncs.com"
    ),
)

# Cache paths — local reads from hydro_setting.yml, cloud from env or default
try:
    from hydrodataset.configs.settings import get_cache_dir
    LOCAL_CACHE = Path(get_cache_dir())
except Exception:
    LOCAL_CACHE = Path(os.getenv("LOCAL_CACHE_DIR", "D:/netcdf"))

CLOUD_CACHE = os.getenv("CLOUD_CACHE_PATH", "camels-us/cache")


# ── Helpers ────────────────────────────────────────────────────────────


def open_nc(path, engine=None):
    """Open a NetCDF file — local path or s3:// URI."""
    kwargs = {}
    if engine:
        kwargs["engine"] = engine
    if str(path).startswith("s3://"):
        fs = s3fs.S3FileSystem(**S3_CONFIG)
        store = fs.get_mapper(path)
        return xr.open_dataset(store, engine="h5netcdf")
    return xr.open_dataset(path, **kwargs)


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

    attr_ds = timed("open attrs nc", lambda: open_nc(attr_path))
    print(f"    dims: {dict(attr_ds.sizes)}")

    def read_p_mean():
        return float(attr_ds[VAR_STATIC].sel(basin=STATION).values)

    v = timed(f"  read {VAR_STATIC} for {STATION}", read_p_mean)
    print(f"    → {VAR_STATIC} = {v:.4f}")

    # ── Dynamic timeseries ──
    print()
    print("═══ Dynamic timeseries ═══")

    ts_ds = timed("open ts nc", lambda: open_nc(ts_path))
    print(f"    dims: {dict(ts_ds.sizes)}")

    def read_streamflow():
        vals = ts_ds["streamflow"].sel(
            basin=STATION, time=slice(T_RANGE[0], T_RANGE[1])
        )
        return float(vals.mean())

    v = timed(f"  read streamflow ({T_RANGE[0]}~{T_RANGE[1]})", read_streamflow)
    print(f"    → mean streamflow = {v:.4f}")

    attr_ds.close()
    ts_ds.close()

    print()
    print("Done.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "local"
    if mode not in ("local", "cloud"):
        print("Usage: uv run python scripts/bench_cloud.py [local|cloud]")
        sys.exit(1)
    main(mode)
