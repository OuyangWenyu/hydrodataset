# -*- coding: utf-8 -*-
"""Benchmark CAMELS-US read speed (local nc vs cloud zarr).

Reads the variable footprint of experiments/train_with_camels_17attributes.py
(7 forcing + streamflow dynamic + 17 static attributes) over a subset: default
100 basins x 5 years. Tune with --basins / --years.

Measures the real read path (read_ts_xrdataset / read_attr_xrdataset -> load),
so it reflects the on-disk / on-OSS chunking. Use as a yardstick for chunk size.

Local nc and cloud zarr live on different machines (nc on the PC, zarr read on
the ECS for internal bandwidth), so run one source per machine.

Usage:
    uv run python scripts/bench_read_speed.py local     # on the PC
    uv run python scripts/bench_read_speed.py cloud     # on the ECS
    uv run python scripts/bench_read_speed.py cloud --basins 100 --years 5 --repeats 5
"""

from __future__ import annotations

import argparse
import statistics
import time

from hydrodataset import resolve_data_path
from hydrodataset.camels_us import CamelsUs

# train_with_camels_17attributes.py: var_t (7) + var_out streamflow, std names.
DYN_TRAIN17 = [
    "precipitation", "daylight_duration", "solar_radiation",
    "temperature_max", "temperature_min", "vapor_pressure",
    "potential_evapotranspiration", "streamflow",
]
# its var_c (17), std names (area_gages2 -> area).
STA_TRAIN17 = [
    "elev_mean", "slope_mean", "area", "frac_forest", "lai_max", "lai_diff",
    "dom_land_cover_frac", "dom_land_cover", "root_depth_50",
    "soil_depth_statsgo", "soil_porosity", "soil_conductivity",
    "max_water_content", "geol_1st_class", "geol_2nd_class",
    "geol_porostiy", "geol_permeability",
]


def _payload_mib(arrays) -> float:
    return sum(a.nbytes for a in arrays) / 1024 / 1024


def _timed(read_fn, repeats: int):
    """Return (cold_time, [warm_times], sample_arrays)."""
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
        v0 = next(iter(raw.data_vars))
        print(f"ts store var '{v0}': shape={raw[v0].shape} "
              f"chunks={raw[v0].encoding.get('chunks')}")
        raw.close()
    except Exception as e:
        print(f"(could not inspect chunks: {e})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", choices=["local", "cloud"])
    p.add_argument("--basins", type=int, default=100,
                   help="number of basins to read (default 100)")
    p.add_argument("--years", type=int, default=5,
                   help="years of timeseries to read from t_range start (default 5)")
    p.add_argument("--repeats", type=int, default=3, help="warm runs (default 3)")
    args = p.parse_args()

    print(f"===== source={args.source}  (train_with_camels_17attributes footprint) =====")
    uri = resolve_data_path("camels_us", source=args.source)
    print(f"URI: {uri}")
    ds = CamelsUs(uri=uri)
    gage_ids = ds.read_object_ids().tolist()[:args.basins]  # subset; not timed
    start = ds.default_t_range[0]
    t_range = [start, f"{int(start[:4]) + args.years - 1}-12-31"]
    print(f"basins={len(gage_ids)}  dynamic={len(DYN_TRAIN17)} vars  "
          f"static={len(STA_TRAIN17)} attrs  t_range={t_range}")
    _inspect_chunks(ds)

    def read_ts():
        out = ds.read_ts_xrdataset(
            gage_id_lst=gage_ids, t_range=t_range, var_lst=DYN_TRAIN17
        ).load()
        return [out[v] for v in out.data_vars]

    def read_attr():
        out = ds.read_attr_xrdataset(
            gage_id_lst=gage_ids, var_lst=STA_TRAIN17
        ).load()
        return [out[v] for v in out.data_vars]

    cold, warm, arrays = _timed(read_ts, args.repeats)
    _report("timeseries", cold, warm, _payload_mib(arrays))
    cold, warm, arrays = _timed(read_attr, args.repeats)
    _report("attributes", cold, warm, _payload_mib(arrays))


if __name__ == "__main__":
    main()
