import os
import unittest

import definitions
from hydrobench.ecmwf4basins.download_era5_land import download_era5


class MyTestCase(unittest.TestCase):
    def test_download_era5(self):
        save_dir = os.path.join(definitions.ROOT_DIR, "test", "test_data")
        downloaded_file = os.path.join(save_dir, "a_test_range.nc")
        date_range = ["2000-01-01", "2000-01-03"]
        lat_lon_range = (31, 108, 30, 109)  # lat_max, lon_min, lat_min, lon_max
        variables_list = 'total_precipitation'
        download_era5(downloaded_file, date_range, lat_lon_range, variables_list, file_format="netcdf")
        print("Downloading ERA5 hourly data is finished!")


if __name__ == '__main__':
    unittest.main()
