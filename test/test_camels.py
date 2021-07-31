import os
import unittest
import definitions
from hydrobench.data.data_camels import Camels


class CamelsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.camels_path = os.path.join(definitions.DATASET_DIR, "camels")

    def test_download_camels(self):
        camels = Camels(self.camels_path, download=True)
        self.assertTrue(os.path.isfile(os.path.join(self.camels_path, "basin_set_full_res", "HCDN_nhru_final_671.shp")))

    def test_download_camels_after_downloading(self):
        camels = Camels(self.camels_path, download=True)
        self.assertTrue(os.path.isfile(os.path.join(self.camels_path, "basin_set_full_res", "HCDN_nhru_final_671.shp")))


if __name__ == '__main__':
    unittest.main()
