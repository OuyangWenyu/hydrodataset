# -*- coding: utf-8 -*-
"""Convert local CAMELS-US NetCDF cache to zarr format.

Usage:
    uv run python scripts/gen_zarr.py

Output:
    scripts/camels_us_attributes.zarr/
    scripts/camels_us_timeseries.zarr/
"""

import xarray as xr
import zarr
import numpy as np

CACHE_DIR = "D:/netcdf"
OUT_DIR = "scripts"


def nc_to_zarr(nc_path, zarr_path, chunk_map):
    """Convert NetCDF to zarr by writing arrays directly, then wrap with xarray metadata."""
    ds = xr.open_dataset(nc_path)
    root = zarr.open_group(zarr_path, mode="w")

    # Write data variables
    for name, da in ds.data_vars.items():
        shp = tuple(int(s) for s in da.shape)
        chk = tuple(chunk_map.get(d, s) for d, s in zip(da.dims, shp))
        arr = root.create_array(name, shape=shp, chunks=chk,
                                dtype=da.dtype, fill_value=None,
                                dimension_names=tuple(da.dims))
        arr[:] = da.values

    # Write dimension coordinate arrays
    for name, da in ds.coords.items():
        if name in ds.dims:
            shp = tuple(int(s) for s in da.shape)
            arr = root.create_array(name, shape=shp, chunks=shp,
                                    dtype=da.dtype, fill_value=None,
                                    dimension_names=(name,))
            arr[:] = da.values
    ds.close()
    print(f"  {nc_path} -> {zarr_path}")

    # Verify: write succeeds, now test read
    ds2 = xr.open_dataset(zarr_path, engine="zarr", consolidated=False)
    print(f"    verify: dims={dict(ds2.sizes)}, vars={len(ds2.data_vars)}")


print("Attributes...")
nc_to_zarr(f"{CACHE_DIR}/camels_us_attributes.nc",
           f"{OUT_DIR}/camels_us_attributes.zarr",
           {"basin": 671})

print("Timeseries...")
nc_to_zarr(f"{CACHE_DIR}/camels_us_timeseries.nc",
           f"{OUT_DIR}/camels_us_timeseries.zarr",
           {"basin": 100, "time": 365})

print("Done.")

print("Done. Upload scripts/*.zarr to s3://camels-us/cache/")
