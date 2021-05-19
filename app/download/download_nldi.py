import os
import sys

from pynhd import NLDI

sys.path.append("../..")
import definitions
from src.data.data_camels import Camels
from src.utils.hydro_utils import progress_wrapped, hydro_logger


@progress_wrapped(estimated_time=671)
def main():
    camels = Camels(os.path.join(definitions.DATASET_DIR, "camels"), download=True)
    basin_id = camels.camels_sites["gauge_id"].values.tolist()
    basins = NLDI().get_basins(basin_id)
    save_dir = os.path.join(definitions.DATASET_DIR, "daymet4basins", "nldi_camels_671_basins")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "nldi_camels_671_basins.shp")
    basins.to_file(save_path)


# python download_nldi.py
if __name__ == '__main__':
    hydro_logger.info("Download the shpfile of 671 basins with GAGE ID in CAMELS")
    main()
