"""Read data from a specified dataset.

Usage:
    python read_dataset.py camels_us
    python read_dataset.py camels_us --source cloud
    python read_dataset.py lamah_ce
"""

import argparse
import importlib

from hydrodataset import resolve_data_path
from hydrodataset.configs.data_resolver import READER_ALIASES, _DEFAULT_REGISTRY


def main():
    available = sorted(_DEFAULT_REGISTRY.keys())

    parser = argparse.ArgumentParser(description="Read data from a specified dataset.")
    parser.add_argument(
        "dataset",
        nargs="?",  # make it optional
        # default="camels_aus",  # change this to test different datasets
        # default="camels_br",
        # default="camels_ch",
        # default="camels_cl",
        # default="camels_col",
        # default="camels_de",
        # default="camels_dk",
        # default="camels_fi",
        # default="camels_fr",
        # default="camels_gb",
        # default="camels_ind",
        # default="camels_lux",
        # default="camels_nz",
        # default="camels_se",
        default="camels_us",
        # default="camelsh_kr",
        # default="camelsh",
        # default="bull",
        # default="caravan_dk",
        # default="hysets",
        # default="estreams",
        # default="lamah_ice",
        # default="simbi",
        # default="lamah_ce",
        # default="grdc_caravan",
        # default="camels",
        help="Name of the dataset to read.",
        choices=available,
    )
    parser.add_argument(
        "--source",
        default="local",
        choices=["local", "cloud"],
        help="Data source: 'local' reads from local path, 'cloud' reads zarr on OSS.",
    )
    args = parser.parse_args()

    alias = READER_ALIASES[args.dataset]
    module_name = alias["module"]
    class_name = alias["class"]

    try:
        module = importlib.import_module(module_name)
        dataset_class = getattr(module, class_name)
    except ImportError:
        print(f"Error: Could not import {class_name} from {module_name}.")
        return

    uri = resolve_data_path(args.dataset, source=args.source)
    print(f"Source: {args.source}  URI: {uri}")

    print(f"Reading from {args.dataset} dataset...")
    ds = dataset_class(uri=uri)

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
