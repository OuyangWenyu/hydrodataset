r"""
Author: Wenyu Ouyang
Date: 2025-10-19 17:09:08
LastEditTime: 2025-10-19 17:09:48
LastEditors: Wenyu Ouyang
Description: Read data from a specified dataset
FilePath: \hydrodataset\examples\read_dataset.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import argparse
import importlib

from hydrodataset import SETTING

DATASET_MAPPING = {
    "camels_aus": ("hydrodataset.camels_aus", "CamelsAus"),
    "camels_br": ("hydrodataset.camels_br", "CamelsBr"),
    "camels_ch": ("hydrodataset.camels_ch", "CamelsCh"),
    "camels_cl": ("hydrodataset.camels_cl", "CamelsCl"),
    "camels_col": ("hydrodataset.camels_col", "CamelsCol"),
    "camels_de": ("hydrodataset.camels_de", "CamelsDe"),
    "camels_dk": ("hydrodataset.camels_dk", "CamelsDk"),
    "camels_fi": ("hydrodataset.camels_fi", "CamelsFi"),
    "camels_fr": ("hydrodataset.camels_fr", "CamelsFr"),
    "camels_gb": ("hydrodataset.camels_gb", "CamelsGb"),
    "camels_ind": ("hydrodataset.camels_ind", "CamelsInd"),
    "camels_lux": ("hydrodataset.camels_lux", "CamelsLux"),
    "camels_nz": ("hydrodataset.camels_nz", "CamelsNz"),
    "camels_se": ("hydrodataset.camels_se", "CamelsSe"),
    "camels_sk": ("hydrodataset.camels_sk", "CamelsSk"),
    "camelsh": ("hydrodataset.camelsh", "Camelsh"),
    "caravan": ("hydrodataset.caravan", "Caravan"),
    "grdc_caravan": ("hydrodataset.grdc_caravan", "GrdcCaravan"),
    "hysets": ("hydrodataset.hysets", "Hysets"),
    "lamah_ce": ("hydrodataset.lamah_ce", "LamahCe"),
    "lamah_ice": ("hydrodataset.lamah_ice", "LamahIce"),
    "mopex": ("hydrodataset.mopex", "Mopex"),
}


def main():
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
        default="camels_ind",
        help="Name of the dataset to read.",
        choices=DATASET_MAPPING.keys(),
    )
    args = parser.parse_args()

    module_name, class_name = DATASET_MAPPING[args.dataset]

    try:
        module = importlib.import_module(module_name)
        dataset_class = getattr(module, class_name)
    except ImportError:
        print(f"Error: Could not import {class_name} from {module_name}.")
        return

    data_path = SETTING["local_data_path"]["datasets-origin"]

    ds = dataset_class(data_path)

    print(f"Reading from {args.dataset} dataset...")

    gage_ids = ds.read_object_ids()
    print("Gage IDs:")
    print(gage_ids)
    print("--------------------------------")

    ts_all = ds.dynamic_features()
    print("All dynamic features:")
    print(ts_all)
    print("--------------------------------")

    attr_all = ds.static_features()
    print("All static features:")
    print(attr_all)
    print("--------------------------------")

    print("Reading timeseries data...")
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=gage_ids[:2],
        t_range=[ds.default_t_range[0], ds.default_t_range[0]],
    )
    print(ts_data)
    print("--------------------------------")

    print("Reading attribute data...")
    attr_data = ds.read_attr_xrdataset(
        gage_id_lst=gage_ids[:2],
        var_lst=attr_all[:2],
    )
    print(attr_data)


if __name__ == "__main__":
    main()
