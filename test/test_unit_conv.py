"""
Author: Wenyu Ouyang
Date: 2023-07-25 20:45:55
LastEditTime: 2023-07-26 16:59:46
LastEditors: Wenyu Ouyang
Description: Test unit conversion
FilePath: \hydrodataset\test\test_unit_conv.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import pandas as pd
import xarray as xr
import pint_xarray
from pint import UnitRegistry
import pint_pandas
import io
import pint


def test_pint_unit_conv():
    unit_to_check = "meter"
    print(f"{unit_to_check} is defined in the unit registry.")

    # Create a DataFrame
    df = pd.DataFrame(
        {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}, index=["01", "02", "03"]
    )
    # Convert the DataFrame to a Dataset
    ds_from_df = df.to_xarray()

    # List of units corresponding to each column
    units = ['meter', 'second', 'kilogram']

    # Assign units to the variables in the Dataset
    for col, unit in zip(df.columns, units):
        ds_from_df[col].attrs['units'] = unit

    ds_from_df = ds_from_df.pint.quantify()
    print(ds_from_df.pint.to({"A": "km"}))

    ds = xr.Dataset(
        {
            "a": (("lon", "lat"), [[11.84, 3.12, 9.7], [7.8, 9.3, 14.72]]),
            "b": (("lon", "lat"), [[13, 2, 7], [5, 4, 9]], {"units": "m"}),
        },
        coords={"lat": [10, 20, 30], "lon": [74, 76]},
    )
    q = ds.pint.quantify(a="s")
    c = q.pint.to({"a": "ms", "b": "km"})
    print(c)
