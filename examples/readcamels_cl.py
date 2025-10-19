"""
Author: Yimeng Zhang
Date: 2025-10-18 17:12:28
LastEditTime: 2025-10-19 11:27:27
LastEditors: Wenyu Ouyang
Description: read CAMELS-CL dataset
FilePath: \hydrodataset\examples\readcamels_cl.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

from hydrodataset.camels_cl import CamelsCl
from hydrodataset import SETTING

data_path = SETTING["local_data_path"]["datasets-origin"]


def main():
    ds = CamelsCl(data_path)
    gage_ids = ds.read_object_ids()
    print(gage_ids)
    print("--------------------------------")
    ts_all = ds.dynamic_features()
    print(ts_all)
    print("--------------------------------")
    attr_all = ds.static_features()
    print(attr_all)
    print("--------------------------------")
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=["3022001"],
        t_range=["2006-07-12", "2006-07-12"],
    )
    print(ts_data)
    print("--------------------------------")
    attr_data = ds.read_attr_xrdataset(
        gage_id_lst=gage_ids[:2],
        var_lst=["elev_mean"],
    )
    print(attr_data)


if __name__ == "__main__":
    main()
