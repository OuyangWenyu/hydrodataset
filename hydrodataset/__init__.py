"""
Author: Wenyu Ouyang
Date: 2022-09-05 23:20:24
LastEditTime: 2022-09-08 14:44:13
LastEditors: Wenyu Ouyang
Description: set file dir
FilePath: \hydrodataset\hydrodataset\__init__.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
from pathlib import Path
import os

__author__ = """Wenyu Ouyang"""
__email__ = "wenyuouyang@outlook.com"
__version__ = '0.0.2'

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
    sys_root_dir = os.path.abspath(os.sep)
    hydrodataset_setting_file.write_text(
        os.path.join(sys_root_dir, "data", "hydrodataset")
    )
hydrodataset_root_dir = Path(hydrodataset_setting_file.read_text())
if not hydrodataset_root_dir.is_dir():
    try:
        hydrodataset_root_dir.mkdir(parents=True)
    except PermissionError:
        print(
            "You cannot create this directory: "
            + hydrodataset_root_dir
            + "\n Please change the first line in "
            + hydrodataset_setting_file
            + " to a directory you have permission and run the code agian"
        )
# set some constants for hydrodataset
ROOT_DIR = hydrodataset_root_dir
CACHE_DIR = hydrodataset_cache_dir
