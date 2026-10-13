# -*- coding: utf-8 -*-
"""Benchmark CAMELSH (hourly) read speed: local nc vs cloud zarr.

Reads an adjustable footprint (variables / years / basins) via the real read
path (read_ts_xrdataset / read_attr_xrdataset -> load), so it reflects the
on-disk / on-OSS chunking. Use as a yardstick for tuning the zarr chunk size.

Local nc (batch files) and cloud zarr live on different machines, so run one
source per machine.

Standard dynamic names (13): streamflow, precipitation, temperature_mean,
  longwave_solar_radiation, solar_radiation, potential_evapotranspiration,
  surface_pressure, u_wind_speed, v_wind_speed, relative_humidity, water_level,
  cape, crainf_frac
Standard static names (5): area, p_mean, p_seasonality, frac_snow, aridity

Usage:
    uv run python scripts/bench_camelsh_read.py local
    uv run python scripts/bench_camelsh_read.py cloud
    uv run python scripts/bench_camelsh_read.py cloud --basins 200 --years 2
    uv run python scripts/bench_camelsh_read.py cloud --vars streamflow,precipitation --attrs all
"""

from __future__ import annotations

import argparse
import statistics
import time

from hydrodataset import resolve_data_path
from hydrodataset.camelsh import Camelsh

DYN_ALL = [
    "streamflow", "precipitation", "temperature_mean",
    "longwave_solar_radiation", "solar_radiation",
    "potential_evapotranspiration", "surface_pressure",
    "u_wind_speed", "v_wind_speed", "relative_humidity",
    "water_level", "cape", "crainf_frac",
]
STA_ALL = ["area", "p_mean", "p_seasonality", "frac_snow", "aridity"]

DYN_DEFAULT = [
    "streamflow", "precipitation", "temperature_mean",
    "potential_evapotranspiration", "solar_radiation",
    "longwave_solar_radiation", "surface_pressure", "relative_humidity",
]


def _payload_mib(arrays) -> float:
    return sum(a.nbytes for a in arrays) / 1024 / 1024


def _timed(read_fn, repeats: int):
    t0 = time.perf_counter()
    arrays = read_fn()
    cold = time.perf_counter() - t0
    warm = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        read_fn()
        warm.append(time.perf_counter() - t0)
    return cold, warm, arrays


def _report(label: str, cold: float, warm: list, mib: float) -> None:
    print(f"\n[{label}] payload={mib:.2f} MiB")
    print(f"  cold: {cold:.3f}s  ({mib / cold:.1f} MiB/s)")
    for i, t in enumerate(warm):
        print(f"  warm {i + 1}: {t:.3f}s  ({mib / t:.1f} MiB/s)")
    best = min(warm)
    print(
        f"  summary: min={best:.3f}s  median={statistics.median(warm):.3f}s  "
        f"mean={statistics.mean(warm):.3f}s  best={mib / best:.1f} MiB/s"
    )


def _inspect_chunks(ds) -> None:
    try:
        raw = ds._load_ts_dataset()
        v0 = next(v for v in raw.data_vars if v != "_progress")
        print(f"ts store var '{v0}': shape={raw[v0].shape} "
              f"chunks={raw[v0].encoding.get('chunks')}")
        raw.close()
    except Exception as e:
        print(f"(could not inspect chunks: {e})")


def _parse_list(value: str, full: list) -> list:
    if value.strip().lower() == "all":
        return full
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", choices=["local", "cloud"])
    p.add_argument("--basins", type=int, default=1000,
                   help="number of basins to read (default 1000)")
    p.add_argument("--years", type=int, default=10,
                   help="years of timeseries from t_range start (default 10)")
    p.add_argument("--vars", default=",".join(DYN_DEFAULT),
                   help="dynamic standard names, comma-separated, or 'all'")
    p.add_argument("--attrs", default=",".join(STA_ALL),
                   help="static standard names, comma-separated, or 'all'")
    p.add_argument("--repeats", type=int, default=3, help="warm runs (default 3)")
    args = p.parse_args()

    dyn = _parse_list(args.vars, DYN_ALL)
    sta = _parse_list(args.attrs, STA_ALL)

    print(f"===== source={args.source}  camelsh (hourly) =====")
    uri = resolve_data_path("camelsh", source=args.source)
    print(f"URI: {uri}")
    ds = Camelsh(uri=uri)
    gage_ids = ds.read_object_ids().tolist()[:args.basins]  # subset; not timed
    start = ds.default_t_range[0]
    t_range = [start, f"{int(start[:4]) + args.years - 1}-12-31"]
    print(f"basins={len(gage_ids)}  dynamic={len(dyn)} vars  "
          f"static={len(sta)} attrs  t_range={t_range}")
    # Cloud only: for local camelsh the store is batch files, and _load_ts_dataset
    # would regenerate them (the single-file cache is absent by design).
    if args.source == "cloud":
        _inspect_chunks(ds)

    def read_ts():
        out = ds.read_ts_xrdataset(
            gage_id_lst=gage_ids, t_range=t_range, var_lst=dyn
        ).load()
        return [out[v] for v in out.data_vars]

    def read_attr():
        out = ds.read_attr_xrdataset(
            gage_id_lst=gage_ids, var_lst=sta
        ).load()
        return [out[v] for v in out.data_vars]

    cold, warm, arrays = _timed(read_ts, args.repeats)
    _report("timeseries", cold, warm, _payload_mib(arrays))
    cold, warm, arrays = _timed(read_attr, args.repeats)
    _report("attributes", cold, warm, _payload_mib(arrays))


if __name__ == "__main__":
    main()
