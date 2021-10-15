import os.path
import unittest

import definitions
from hydrobench.nldas4basins.download_nldas import download_nldas_with_url_lst


class MyTestCase(unittest.TestCase):
    def test_download_nldas_hourly(self):
        url_lst_file = os.path.join(definitions.ROOT_DIR, "hydrobench", "nldas4basins",
                                    "subset_NLDAS_FORA0125_H_2.0_20211015_140101.txt")
        save_dir = os.path.join(definitions.DATASET_DIR, "nldas_hourly")
        download_nldas_with_url_lst(url_lst_file, save_dir)
        print("Downloading NLDAS hourly data is finished!")


if __name__ == '__main__':
    unittest.main()
