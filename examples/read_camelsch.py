"""
Author: Yimeng Zhang
Date: 2025-10-18 17:12:28
LastEditTime: 2025-10-19 10:47:41
LastEditors: Wenyu Ouyang
Description: read CAMELS-CH dataset
FilePath: \hydrodataset\examples\readcamels_ch.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

from hydrodataset.camels_ch import CamelsCh
from hydrodataset import SETTING

data_path = SETTING["local_data_path"]["datasets-origin"]


def main():
    ds = CamelsCh(data_path)
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
        gage_id_lst=["2349"],
        t_range=["1981-01-01", "1981-01-01"],
    )
    print(ts_data)
    print("--------------------------------")
    attr_data = ds.read_attr_xrdataset(
        gage_id_lst=gage_ids[:2],
        var_lst=["p_mean"],
    )
    print(attr_data)


if __name__ == "__main__":
    main()
