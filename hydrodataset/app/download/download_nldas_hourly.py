"""
Author: Wenyu Ouyang
Date: 2022-01-25 16:49:00
LastEditTime: 2022-03-19 16:37:33
LastEditors: Wenyu Ouyang
Description: Download Hourly NLDAS data
FilePath: /HydroBench/hydrodataset/app/download/download_nldas_hourly.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import argparse
import os
import sys

from pathlib import Path

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent.parent))
import definitions
from hydrodataset.utils.hydro_utils import hydro_logger
from hydrodataset.nldas4basins.download_nldas import download_nldas_with_url_lst


def main(args):
    download_lst_dir = os.path.join(definitions.ROOT_DIR, "hydrobench", "nldas4basins")
    save_dir = os.path.join(definitions.DATASET_DIR, "nldas_hourly", str(args.year))
    for file in os.listdir(download_lst_dir):
        if "NLDAS" in file and ".txt" in file:
            url_lst_file = os.path.join(
                definitions.ROOT_DIR, "hydrobench", "nldas4basins", file
            )
            download_nldas_with_url_lst(url_lst_file, save_dir)
    print("Downloading NLDAS hourly data is finished!")


# python download_nldas_hourly.py --year 2014
if __name__ == "__main__":
    hydro_logger.info("Download the NLDAS hourly forcing data for CONUS!")
    parser = argparse.ArgumentParser(description="Download NLDAS hourly data")
    parser.add_argument(
        "--year",
        dest="year",
        help="The year of downloaded data",
        default=2014,
        type=int,
    )
    the_args = parser.parse_args()
    main(the_args)
