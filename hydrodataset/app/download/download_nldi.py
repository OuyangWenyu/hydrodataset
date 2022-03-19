"""
Author: Wenyu Ouyang
Date: 2022-01-25 16:49:00
LastEditTime: 2022-03-19 16:37:59
LastEditors: Wenyu Ouyang
Description: Download basins' shapefile from NLDI
FilePath: /HydroBench/hydrodataset/app/download/download_nldi.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import os
import sys

from pynhd import NLDI

from pathlib import Path

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent.parent))
import definitions
from hydrodataset.data.data_camels import Camels
from hydrodataset.utils.hydro_utils import progress_wrapped, hydro_logger


@progress_wrapped(estimated_time=1000)
def main():
    camels = Camels(os.path.join(definitions.DATASET_DIR, "camels", "camels_us"))
    basin_id = camels.camels_sites["gauge_id"].values.tolist()
    basins = NLDI().get_basins(basin_id)
    save_dir = os.path.join(
        definitions.DATASET_DIR, "daymet4camels", "nldi_camels_671_basins"
    )
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "nldi_camels_671_basins.shp")
    basins.to_file(save_path)


# python download_nldi.py
if __name__ == "__main__":
    hydro_logger.info("Download the shpfile of 671 basins with GAGE ID in CAMELS")
    main()
