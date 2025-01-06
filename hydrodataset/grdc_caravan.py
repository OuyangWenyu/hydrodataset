"""
Author: Wenyu Ouyang
Date: 2024-12-30 18:44:19
LastEditTime: 2025-01-06 08:23:02
LastEditors: Wenyu Ouyang
Description: For GRDC-Caravan dataset
FilePath: \hydrodataset\hydrodataset\grdc_caravan.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import collections
import os
import tarfile
import warnings
import tqdm
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
        the_dict["DOWNLOAD_URL"] = [
            "https://zenodo.org/records/14006282/files/caravan-grdc-extension-csv.tar.gz",
            "https://zenodo.org/records/14006282/files/caravan-grdc-extension-nc.tar.gz",
            "https://zenodo.org/records/14006282/files/grdc-caravan_data_description.pdf",
        ]
        the_dict["TS_CSV_DIR"] = os.path.join(
            self.data_source_dir,
            "GRDC-Caravan-extension-csv",
            "timeseries",
            "csv",
        )
        return the_dict

    def _base_dir(self):
        # we use csv directory to read the data
        return os.path.join(self.data_source_dir, "GRDC-Caravan-extension-nc")

    def download_data_source(self) -> None:
        """
        Download dataset.

        Returns
        -------
        None
        """
        self.data_source_dir.mkdir(exist_ok=True)
        print(
            "We only support manual downloading now. Please download two tar.gz files and one pdf file from the download links below:"
            + "https://zenodo.org/records/14006282/files/caravan-grdc-extension-csv.tar.gz\n"
            + "https://zenodo.org/records/14006282/files/caravan-grdc-extension-nc.tar.gz \n"
            + "https://zenodo.org/records/14006282/files/grdc-caravan_data_description.pdf"
        )
        for zipfile in self.data_source_description["DOWNLOAD_URL"][:-1]:
            file_name = os.path.join(self.data_source_dir, zipfile.split("/")[-1])
            try:
                with tarfile.open(file_name, "r:gz") as tar:
                    # Create a tqdm progress bar, assuming the total number of files equals the number of members in the tar archive (this may not be accurate).
                    with tqdm.tqdm(
                        total=len(tar.getmembers()), desc="Extracting"
                    ) as pbar:
                        for member in tar.getmembers():
                            # We are not actually extracting each file to update the progress bar, because that would be too slow. Instead, we simulate progress updates (which does not reflect actual progress).
                            tar.extract(
                                member, path=self.data_source_dir
                            )  # extract the file
                            pbar.update(1)  # update the progress bar
            except tarfile.ReadError:
                warnings.warn("Please manually unzip the file.")
