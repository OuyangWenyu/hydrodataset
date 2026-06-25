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

# Cache paths — local reads from hydro_setting.yml, cloud from env or default
try:
    from hydrodataset.configs.settings import get_cache_dir
    LOCAL_CACHE = Path(get_cache_dir())
except Exception:
    LOCAL_CACHE = Path(os.getenv("LOCAL_CACHE_DIR", "D:/netcdf"))

CLOUD_CACHE = os.getenv("CLOUD_CACHE_PATH", "camels-us/cache")


# ── Helpers ────────────────────────────────────────────────────────────


def get_s3fs():
    import s3fs
    return s3fs.S3FileSystem(
        key=os.environ["OSS_ACCESS_KEY_ID"],
        secret=os.environ["OSS_ACCESS_KEY_SECRET"],
        client_kwargs={"region_name": os.environ.get("OSS_REGION", "cn-beijing")},
        config_kwargs={"s3": {"addressing_style": "virtual"}},
        endpoint_url=os.environ.get(
            "OSS_ENDPOINT", "https://oss-cn-beijing-internal.aliyuncs.com"
        ),
    )


def open_ds(path, fmt="nc"):
    """Open a dataset — nc or zarr, local or s3://."""
    if fmt == "zarr" and str(path).startswith("s3://"):
        return xr.open_dataset(path, engine="zarr", storage_options=dict(
            key=os.environ["OSS_ACCESS_KEY_ID"],
            secret=os.environ["OSS_ACCESS_KEY_SECRET"],
            client_kwargs={"region_name": os.environ.get("OSS_REGION", "cn-beijing")},
            config_kwargs={"s3": {"addressing_style": "virtual"}},
            endpoint_url=os.environ.get("OSS_ENDPOINT", "https://oss-cn-beijing-internal.aliyuncs.com"),
        ), chunks={})
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
