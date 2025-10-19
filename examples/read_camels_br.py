from hydrodataset.camels_br import CamelsBr
from hydrodataset import SETTING

data_path = SETTING["local_data_path"][
    "datasets-origin"
]  # change the path in hydro_setting.yml in your user folder


def main():
    # data_path is the parent directory of the dataset
    ds = CamelsBr(data_path)
    gage_ids = ds.read_object_ids()
    print(gage_ids)
    print("--------------------------------")
    ts_all = ds.dynamic_features()
    print(ts_all)
    print("--------------------------------")
    attr_all = ds.static_features()
    print(attr_all)
    print("--------------------------------")
    dx = ds.read_ts_xrdataset(
        gage_id_lst=gage_ids[:1],
        t_range=["1980-01-01", "1980-01-01"],
        var_lst=["p_mswep", "streamflow_m3s"],
    )
    print(dx)
    print("--------------------------------")
    dy = ds.read_attr_xrdataset(
        gage_id_lst=gage_ids[:2],
        var_lst=["p_mean"],
    )
    print(dy)


if __name__ == "__main__":
    main()
