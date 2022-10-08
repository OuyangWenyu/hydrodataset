"""
Author: Wenyu Ouyang
Date: 2022-09-05 23:20:24
LastEditTime: 2022-10-08 14:23:27
LastEditors: Wenyu Ouyang
Description: set file dir
FilePath: \hydrodataset\hydrodataset\__init__.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
from pathlib import Path
import os

__author__ = """Wenyu Ouyang"""
__email__ = "wenyuouyang@outlook.com"
__version__ = '0.0.8'

# we use a .hydrodataset dir to save the setting
# more file/dir operations could be seen in:
# https://zhuanlan.zhihu.com/p/139783331
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
from .hydro_dataset import *
from .camels import *
