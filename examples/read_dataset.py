"""Read data from a specified dataset via the unified open_dataset factory.

Usage:
    python read_dataset.py camels_us
    python read_dataset.py lamah_ce
"""

import argparse

from hydrodataset import open_dataset
from hydrodataset.configs.data_resolver import _DEFAULT_REGISTRY


def main():
    available = sorted(_DEFAULT_REGISTRY.keys())

    parser = argparse.ArgumentParser(description="Read data from a specified dataset.")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="lamah_ce",
        help="Name of the dataset to read.",
        choices=available,
    )
    args = parser.parse_args()

    print(f"Reading from {args.dataset} dataset...")
    ds = open_dataset(args.dataset)

    gage_ids = ds.read_object_ids()
    print("Gage IDs:")
    print(gage_ids)
    print(f"Number of gages: {len(gage_ids)}")
    print("--------------------------------")

    t_range = ds.default_t_range
    print(f"Default time range: {t_range}")
    print("--------------------------------")

    ts_available = ds.available_dynamic_features
    print("Available dynamic features:")
    print(ts_available)
    print("--------------------------------")

    attr_available = ds.available_static_features
    print("Available static features:")
    print(attr_available)
    print("--------------------------------")

    print("Reading timeseries data...")
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=gage_ids[-1:],
        t_range=[ds.default_t_range[0], ds.default_t_range[0]],
        var_lst=["precipitation", "streamflow"],
    )
    print(ts_data)
    print("--------------------------------")

    print("Reading attribute data...")
    attr_data = ds.read_attr_xrdataset(
        gage_id_lst=gage_ids[:2],
        var_lst=["area"],
    )
    print(attr_data)

    print("\nTesting read_area...")
    area = ds.read_area(gage_id_lst=gage_ids[:2])
    print(area)
    print("--------------------------------")

    print("\nTesting read_mean_prcp...")
    mean_prcp = ds.read_mean_prcp(gage_id_lst=gage_ids[:2])
    print(mean_prcp)
    print("--------------------------------")


if __name__ == "__main__":
    main()
