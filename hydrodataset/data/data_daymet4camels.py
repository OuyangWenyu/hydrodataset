import collections
import glob
import logging
import os
from typing import Union, List

import pandas as pd
import numpy as np
import xarray as xr

from hydrodataset.data.data_base import DataSourceBase
from hydrodataset.data.data_camels import Camels
from hydrodataset.utils import hydro_utils


class Daymet4Camels(DataSourceBase):
    """
    A datasource class for geo attributes data, Daymet v4 forcing data, and streamflow data of basins in CAMELS.

    The forcing data could be gridded data, or basin mean values. Attributes and streamflow data come from CAMELS.
    """

    def __init__(self, data_path, camels_data_path, download=False):
        """
        Initialize a daymet4basins instance.

        Parameters
        ----------
        data_path
            the data file directory for the instance
        camels_data_path
            CAMELS will be used here, so its path should be provided
        download
            if True we will download the gridded data

        """
        super().__init__(data_path)
        self.camels = Camels(camels_data_path)
        self.data_source_description = self.set_data_source_describe()
        if download:
            self.download_data_source()
        self.camels671_sites = self.read_site_info()

    def get_name(self):
        return "DAYMET4CAMELS"

    def set_data_source_describe(self):
        daymet_db = self.data_source_dir
        # shp file of basins
        camels671_shp = self.camels.data_source_description["CAMELS_BASINS_SHP_FILE"]
        # shp file of basins from NLDI. It is different with that in CAMELS, so now we didn't use it generally.
        camels671_shp_from_nldi = os.path.join(
            daymet_db, "nldi_camels_671_basins", "nldi_camels_671_basins.shp"
        )
        # forcing
        forcing_original_dir = os.path.join(daymet_db, "daymet_camels_671_unmask")
        forcing_dir = os.path.join(daymet_db, "daymet_camels_671_bound")
        forcing_resample_dir = os.path.join(
            daymet_db, "daymet_camels_671_bound_resample"
        )
        forcing_basin_mean_dir = os.path.join(daymet_db, "basin_mean_forcing")

        return collections.OrderedDict(
            DAYMET4BASINS_DIR=daymet_db,
            CAMELS_SHP_FILE=camels671_shp,
            NLDI_SHP_FILE=camels671_shp_from_nldi,
            DAYMET4_ORIGIN_DIR=forcing_original_dir,
            DAYMET4_DIR=forcing_dir,
            DAYMET4_RESAMPLE_DIR=forcing_resample_dir,
            DAYMET4_BASIN_MEAN_DIR=forcing_basin_mean_dir,
        )

    def download_data_source(self):
        logging.warning(
            "The data files are too large. Please use HydroBench to download them!"
        )

    def get_constant_cols(self) -> np.array:
        return self.camels.get_constant_cols()

    def get_relevant_cols(self) -> np.array:
        return np.array(
            ["dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp", "petpt", "petfao56"]
        )

    def get_target_cols(self) -> np.array:
        return self.camels.get_target_cols()

    def get_other_cols(self) -> dict:
        pass

    def read_site_info(self) -> pd.DataFrame:
        return self.camels.read_site_info()

    def read_object_ids(self, object_params=None) -> np.array:
        return self.camels.read_object_ids()

    def read_target_cols(
        self, usgs_id_lst=None, t_range=None, target_cols=None, **kwargs
    ):
        return self.camels.read_target_cols(usgs_id_lst, t_range, target_cols)

    def read_basin_mean_daymet4(self, usgs_id, var_lst, t_range_list):
        print("reading %s forcing data", usgs_id)
        gage_id_df = self.camels671_sites
        huc = gage_id_df[gage_id_df["gauge_id"] == usgs_id]["huc_02"].values[0]

        data_folder = self.data_source_description["DAYMET4_BASIN_MEAN_DIR"]
        data_file = os.path.join(
            data_folder, "daymet", huc, "%s_lump_cida_forcing_leap_pet.txt" % usgs_id
        )
        data_temp = pd.read_csv(data_file, sep=r"\s+", header=None, skiprows=1)
        forcing_lst = [
            "Year",
            "Mnth",
            "Day",
            "Hr",
            "dayl",
            "prcp",
            "srad",
            "swe",
            "tmax",
            "tmin",
            "vp",
            "petpt",
            "petfao56",
        ]
        df_date = data_temp[[0, 1, 2]]
        df_date.columns = ["year", "month", "day"]
        date = pd.to_datetime(df_date).values.astype("datetime64[D]")

        nf = len(var_lst)
        [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
        nt = c.shape[0]
        out = np.empty([nt, nf])

        for k in range(nf):
            ind = forcing_lst.index(var_lst[k])
            out[:, k] = data_temp[ind].values[ind1]
        return out

    def read_relevant_cols(
        self,
        usgs_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        concat: bool = False,
        resample: Union[int, float] = 1,
    ) -> Union[xr.Dataset, List[xr.Dataset], np.array]:
        """
        Read forcing data.

        Parameters
        ----------
        usgs_id_lst
            the ids of gages in CAMELS
        t_range
            the start and end periods
        var_lst
            the forcing var types
        concat
            concat means we concat all data_source to one. If concat, return a Xarray Dataset, else return list
        resample
            resample==1: we use the original gridded daymet data;
            resample==0: average data in a basin;
            resample==other ints: coarse the original data with resample times;
            resample==floats: interpolate the original data with resample times;

        Returns
        -------
        Union[xr.Dataset, list, np.array]
            if resample ==1, return an np.array;
            else: if concat return xr.dataset otherwise return list
        """

        assert len(t_range) == 2
        assert all(x < y for x, y in zip(usgs_id_lst, usgs_id_lst[1:]))
        if resample > 0:
            t_years = hydro_utils.t_range_years(t_range)
            # our range is a left open left close range, the default range in xarray slice is close interval, so -1 day
            t_days = hydro_utils.t_days_lst2range(hydro_utils.t_range_days(t_range))
            if resample == 1:
                data_folder = self.data_source_description["DAYMET4_DIR"]
                resample_str = ""
            else:
                data_folder = self.data_source_description["DAYMET4_RESAMPLE_DIR"]
                resample_str = str(resample)
            ens_list = []
            for num in range(len(usgs_id_lst)):
                ens = usgs_id_lst[num]
                name_lst = []
                for name in glob.glob(
                    os.path.join(data_folder, ens, "*" + resample_str + ".nc")
                ):
                    if int(name.split("/")[-1].split("_")[1]) in t_years:
                        name_lst.append(name)
                name_lst_sorted = np.sort(name_lst).tolist()
                ens_list.append(
                    xr.open_mfdataset(name_lst_sorted).sel(
                        time=slice(t_days[0], t_days[1])
                    )
                )

            if concat:
                new_dim = pd.Index(usgs_id_lst)
                ds = xr.concat(ens_list, dim=new_dim)
                return ds
            else:
                return ens_list
        else:
            t_range_list = hydro_utils.t_range_days(t_range)
            nt = t_range_list.shape[0]
            x = np.empty([len(usgs_id_lst), nt, len(var_lst)])
            for k in range(len(usgs_id_lst)):
                data = self.read_basin_mean_daymet4(
                    usgs_id_lst[k], var_lst, t_range_list
                )
                x[k, :, :] = data
            return x

    def read_constant_cols(self, usgs_id_lst=None, var_lst=None, is_return_dict=False):
        return self.camels.read_constant_cols(usgs_id_lst, var_lst, is_return_dict)

    def read_other_cols(self, object_ids=None, other_cols: dict = None, **kwargs):
        pass
