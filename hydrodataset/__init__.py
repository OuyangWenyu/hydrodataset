"""
Author: Wenyu Ouyang
Date: 2022-09-05 23:20:24
LastEditTime: 2025-10-30 11:13:29
LastEditors: Wenyu Ouyang
Description: set file dir
FilePath: \hydrodataset\hydrodataset\__init__.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import os
import yaml
from pathlib import Path
from hydroutils import hydro_file

__author__ = """Wenyu Ouyang"""
__email__ = "wenyuouyang@outlook.com"
__version__ = "0.1.13"


SETTING_FILE = os.path.join(Path.home(), "hydro_setting.yml")


def read_setting(setting_path):
    if not os.path.exists(setting_path):
        raise FileNotFoundError(f"Configuration file not found: {setting_path}")

    with open(setting_path, "r") as file:
        setting = yaml.safe_load(file)

    example_setting = (
        "local_data_path:\n"
        "  root: 'D:\\data\\waterism' # Update with your root data directory\n"
        "  datasets-origin: 'D:\\data\\waterism\\datasets-origin'\n"
    )

    if setting is None:
        raise ValueError(
            f"Configuration file is empty or has invalid format.\n\nExample configuration:\n{example_setting}"
        )

    # Define the expected structure
    expected_structure = {
        "local_data_path": ["root", "datasets-origin"],
    }

    # Validate the structure
    try:
        for key, subkeys in expected_structure.items():
            if key not in setting:
                raise KeyError(f"Missing required key in config: {key}")

            if isinstance(subkeys, list):
                for subkey in subkeys:
                    if subkey not in setting[key]:
                        raise KeyError(f"Missing required subkey '{subkey}' in '{key}'")
    except KeyError as e:
        raise ValueError(
            f"Incorrect configuration format: {e}\n\nExample configuration:\n{example_setting}"
        ) from e

    return setting


try:
    SETTING = read_setting(SETTING_FILE)
    # set some constants for hydrodataset
    ROOT_DIR = SETTING["local_data_path"]["datasets-origin"]
    # As hydrodataset has a lot of datasets, maybe disk C of user is not enough, so we set the cache directory here.
    CACHE_DIR = SETTING["local_data_path"]["cache"]
except ValueError as e:
    print(f"Warning: {e}")
    # Set default values for CI/testing environments
    SETTING = None
    ROOT_DIR = os.path.join(Path.home(), "hydrodataset_data", "datasets-origin")
    CACHE_DIR = os.path.join(Path.home(), "hydrodataset_data", "cache")
except Exception as e:
    print(f"Unexpected error: {e}")
    # Set default values for CI/testing environments
    SETTING = None
    ROOT_DIR = os.path.join(Path.home(), "hydrodataset_data", "datasets-origin")
    CACHE_DIR = os.path.join(Path.home(), "hydrodataset_data", "cache")

# set some constants for datasets
# deprecated Constant for datasets in the near future
DATASETS = ["CAMELS", "Caravan", "GRDC", "HYSETS", "LamaH", "MOPEX"]
CAMELS_REGIONS = ["AUS", "BR", "CL", "GB", "US"]
LAMAH_REGIONS = ["CE"]
# For CANOPEX, We don't treat it as a dataset, but a special case for MOPEX. We only have CANOPEX now.
MOPEX_REGIONS = ["CA"]
REGIONS = CAMELS_REGIONS + LAMAH_REGIONS + MOPEX_REGIONS
from .hydro_dataset import *
from .camels import *
from .camelsh_kr import *
from .multi_datasets import *
from .lamah_ce import *
from .camels_us import *
