"""
Author: Wenyu Ouyang
Date: 2024-12-30 18:44:19
LastEditTime: 2024-12-30 20:26:06
LastEditors: Wenyu Ouyang
Description: For GRDC-Caravan dataset
FilePath: \hydrodataset\hydrodataset\grdc_caravan.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import collections
import os
import tarfile
from hydroutils import hydro_file
from hydrodataset.caravan import Caravan


class GrdcCaravan(Caravan):
    def __init__(self, data_path, download=False):
        """
        Initialization for GRDC-Caravan dataset

        Parameters
        ----------
        data_path
            where we put the dataset
        download
            if true, download
        """
        super().__init__(data_path, download=download, region="GRDC")

    @property
    def region_name_dict(self):
        _region_name_dict = super().region_name_dict
        _region_name_dict["GRDC"] = "grdc"
        return _region_name_dict

    def get_name(self):
        return "GRDC-Caravan"

    def set_data_source_describe(self) -> collections.OrderedDict:
        """
        Introduce the files in the dataset and list their location in the file system

        Returns
        -------
        collections.OrderedDict
            the description for GRDC-Caravan
        """
        the_dict = super().set_data_source_describe()
        # Here we use nc files
        the_dict["DOWNLOAD_URL"] = (
            "https://zenodo.org/record/14006282/files/caravan-grdc-extension-nc.tar.gz?download=1"
        )
        return the_dict

    def _base_dir(self):
        return os.path.join(self.data_source_dir, "GRDC-Caravan")

    def download_data_source(self) -> None:
        """
        Download dataset.

        Returns
        -------
        None
        """
        self.data_source_dir.mkdir(exist_ok=True)
        print(
            "We only support manual downloading now. "
            + "As NetCDF version is enough, we just download each tar.gz file from the following links:"
            + "https://zenodo.org/records/14006282/files/caravan-grdc-extension-csv.tar.gz?download=1"
        )
        file_name = self.data_source_description["DOWNLOAD_URL"].split("/")[-1]
        try:
            with tarfile.open(file_name, "r:gz") as tar:
                # unzip the file
                tar.extractall()
        except tarfile.ReadError:
            Warning("Please manually unzip the file.")
