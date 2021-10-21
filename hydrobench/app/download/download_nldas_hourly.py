import os
import sys

sys.path.append(os.path.join("..", "..", ".."))
import definitions
from hydrobench.utils.hydro_utils import hydro_logger
from hydrobench.nldas4basins.download_nldas import download_nldas_with_url_lst


def main():
    download_lst_dir = os.path.join(definitions.ROOT_DIR, "hydrobench", "nldas4basins")
    save_dir = os.path.join(definitions.DATASET_DIR, "nldas_hourly")
    for file in os.listdir(download_lst_dir):
        if "NLDAS" in file and ".txt" in file:
            url_lst_file = os.path.join(definitions.ROOT_DIR, "hydrobench", "nldas4basins", file)
            download_nldas_with_url_lst(url_lst_file, save_dir)
    print("Downloading NLDAS hourly data is finished!")


# python download_nldas_hourly.py
if __name__ == '__main__':
    hydro_logger.info("Download the NLDAS hourly forcing data for CONUS!")
    main()
