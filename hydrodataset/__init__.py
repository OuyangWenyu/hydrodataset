"""
Author: Wenyu Ouyang
Date: 2022-09-05 23:20:24
LastEditTime: 2023-07-25 15:52:16
LastEditors: Wenyu Ouyang
Description: set file dir
FilePath: \hydrodataset\hydrodataset\__init__.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
from pathlib import Path
import os

__author__ = """Wenyu Ouyang"""
__email__ = "wenyuouyang@outlook.com"
__version__ = '0.1.7'

# we use a .hydrodataset dir to save the setting
hydrodataset_setting_dir = Path.home().joinpath(".hydrodataset")
if not hydrodataset_setting_dir.is_dir():
    hydrodataset_setting_dir.mkdir(parents=True)
hydrodataset_cache_dir = hydrodataset_setting_dir.joinpath("cache")
if not hydrodataset_cache_dir.is_dir():
    hydrodataset_cache_dir.mkdir(parents=True)
hydrodataset_setting_file = hydrodataset_setting_dir.joinpath("settings.txt")
if not hydrodataset_setting_file.is_file():
    hydrodataset_setting_file.touch(exist_ok=False)
    # default data dir is cache, user should modify it to his/her own
    hydrodataset_setting_file.write_text(hydrodataset_cache_dir._str)
# read first line
hydrodataset_root_dir = Path(hydrodataset_setting_file.read_text().split("\n")[0])
try:
    if not os.path.isdir(hydrodataset_root_dir):
        hydrodataset_root_dir.mkdir(parents=True)
except PermissionError:
    print(
        "You cannot create this directory: "
        + hydrodataset_root_dir._str
        + "\nPlease change the first line in "
        + hydrodataset_setting_file._str
        + " to a directory you have permission and run the code agian"
    )
# set some constants for hydrodataset
ROOT_DIR = hydrodataset_root_dir
CACHE_DIR = hydrodataset_cache_dir

# set some constants for datasets
DATASETS = ["CAMELS", "Caravan", "GRDC", "HYSETS", "LamaH", "MOPEX"]
CAMELS_REGIONS = ["AUS", "BR", "CL", "GB", "US"]
LAMAH_REGIONS = ["CE"]
# For CANOPEX, We don't treat it as a dataset, but a special case for MOPEX. We only have CANOPEX now.
MOPEX_REGIONS = ["CA"]
REGIONS = CAMELS_REGIONS + LAMAH_REGIONS + MOPEX_REGIONS
from .hydro_dataset import *
from .camels import *
from .multi_datasets import *
from .lamah import *
