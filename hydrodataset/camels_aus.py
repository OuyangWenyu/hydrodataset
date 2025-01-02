import os
import collections
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
from pandas.api.types import is_string_dtype, is_numeric_dtype
from tqdm import tqdm
from hydroutils import hydro_time, hydro_file
from hydrodataset import CACHE_DIR, HydroDataset, CAMELS_REGIONS
from hydrodataset.camels import Camels

CAMELS_NO_DATASET_ERROR_LOG = (
    "We cannot read this dataset now. Please check if you choose correctly:\n"
    + str(CAMELS_REGIONS)
)


class CamelsAus(Camels):
    def __init__(
        self,
        data_path=os.path.join("camels", "camels_aus"),
        download=False,
        region: str = "AUS",
    ):
        """
        Initialization for CAMELS-AUS dataset

        Parameters
        ----------
        data_path
            where we put the dataset.
            we already set the ROOT directory for hydrodataset,
            so here just set it as a relative path,
            by default "camels/camels_aus"
        download
            if true, download, by default False
        region
            the default is CAMELS-AUS
        """
        super().__init__(data_path, download, region)

    def set_data_source_describe(self) -> collections.OrderedDict:
        """
        the files in the dataset and their location in file system

        Returns
        -------
        collections.OrderedDict
            the description for a CAMELS-AUS dataset
        """
        camels_db = self.data_source_dir

        if self.region == "AUS":
            return self._set_data_source_camelsaus_describe(camels_db)
        elif self.region == "AUS_v2":
            return self._set_data_source_camelsausv2_describe(camels_db)
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)

    def _set_data_source_camelsausv2_describe(self, camels_db):
        # id and name
        gauge_id_file = camels_db.joinpath(
            "01_id_name_metadata",
            "01_id_name_metadata",
            "id_name_metadata.csv",
        )
        # shp file of basins
        camels_shp_file = camels_db.joinpath(
            "02_location_boundary_area",
            "02_location_boundary_area",
            "shp",
            "CAMELS_AUS_v2_BasinOutlets_adopted.shp",
        )
        # config of flow data
        flow_dir = camels_db.joinpath("03_streamflow", "03_streamflow")
        # attr
        attr_dir = camels_db.joinpath("04_attributes", "04_attributes")
        # forcing
        forcing_dir = camels_db.joinpath("05_hydrometeorology", "05_hydrometeorology")

        return collections.OrderedDict(
            CAMELS_DIR=camels_db,
            CAMELS_FLOW_DIR=flow_dir,
            CAMELS_FORCING_DIR=forcing_dir,
            CAMELS_ATTR_DIR=attr_dir,
            CAMELS_GAUGE_FILE=gauge_id_file,
            CAMELS_BASINS_SHP_FILE=camels_shp_file,
        )

    def _set_data_source_camelsaus_describe(self, camels_db):
        # id and name
        gauge_id_file = camels_db.joinpath(
            "01_id_name_metadata",
            "01_id_name_metadata",
            "id_name_metadata.csv",
        )
        # shp file of basins
        camels_shp_file = camels_db.joinpath(
            "02_location_boundary_area",
            "02_location_boundary_area",
            "shp",
            "CAMELS_AUS_BasinOutlets_adopted.shp",
        )
        # config of flow data
        flow_dir = camels_db.joinpath("03_streamflow", "03_streamflow")
        # attr
        attr_dir = camels_db.joinpath("04_attributes", "04_attributes")
        # forcing
        forcing_dir = camels_db.joinpath("05_hydrometeorology", "05_hydrometeorology")

        return collections.OrderedDict(
            CAMELS_DIR=camels_db,
            CAMELS_FLOW_DIR=flow_dir,
            CAMELS_FORCING_DIR=forcing_dir,
            CAMELS_ATTR_DIR=attr_dir,
            CAMELS_GAUGE_FILE=gauge_id_file,
            CAMELS_BASINS_SHP_FILE=camels_shp_file,
        )

    def read_site_info(self) -> pd.DataFrame:
        """
        Read the basic information of gages in a CAMELS-AUS dataset

        Returns
        -------
        pd.DataFrame
            basic info of gages
        """
        camels_file = self.data_source_description["CAMELS_GAUGE_FILE"]
        if self.region in ["AUS", "AUS_v2"]:
            data = pd.read_csv(camels_file, sep=",", dtype={"station_id": str})
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)
        return data

    def get_constant_cols(self) -> np.ndarray:
        """
        all readable attrs in CAMELS-AUS

        Returns
        -------
        np.array
            attribute types
        """
        attr_all_file = os.path.join(
            self.data_source_description["CAMELS_DIR"],
            "CAMELS_AUS_Attributes-Indices_MasterTable.csv",
        )
        camels_aus_attr_indices_data = pd.read_csv(attr_all_file, sep=",")
        # exclude station id
        return camels_aus_attr_indices_data.columns.values[1:]

    def get_relevant_cols(self) -> np.ndarray:
        """
        all readable forcing types in CAMELS-AUS

        Returns
        -------
        np.array
            forcing types
        """
        if self.region == "AUS":
            forcing_types = []
            for root, dirs, files in os.walk(
                self.data_source_description["CAMELS_FORCING_DIR"]
            ):
                if root == self.data_source_description["CAMELS_FORCING_DIR"]:
                    continue
                forcing_types.extend(
                    file[:-4] for file in files if file != "ClimaticIndices.csv"
                )
            the_cols = np.array(forcing_types)
        elif self.region == "AUS_v2":
            forcing_types = []
            for root, dirs, files in os.walk(
                self.data_source_description["CAMELS_FORCING_DIR"]
            ):
                if root == self.data_source_description["CAMELS_FORCING_DIR"]:
                    continue
                forcing_types.extend(
                    file[:-4]
                    for file in files
                    if file not in ["ClimaticIndices.csv", "desktop.ini"]
                )
            the_cols = np.array(forcing_types)
        else:
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)
        return the_cols

    def get_target_cols(self) -> np.ndarray:
        """
        For CAMELS-AUS, the target vars are streamflows

        QualityCodes are not streamflow data.
        MLd means "1 Megaliters Per Day"; 1 MLd = 0.011574074074074 cubic-meters-per-second
        mmd means "mm/day"

        Returns
        -------
        np.array
            streamflow types
        """
        return np.array(
            [
                "streamflow_MLd",
                "streamflow_MLd_inclInfilled",
                "streamflow_mmd",
                "streamflow_QualityCodes",
            ]
        )

    def read_object_ids(self, **kwargs) -> np.ndarray:
        """
        read station ids

        Parameters
        ----------
        **kwargs
            optional params if needed

        Returns
        -------
        np.array
            gage/station ids
        """
        return self.sites["station_id"].values

    def read_target_cols(
        self,
        gage_id_lst: Union[list, np.array] = None,
        t_range: list = None,
        target_cols: Union[list, np.array] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        read target values; for CAMELS-AUS, they are streamflows

        default target_cols is an one-value list
        Notice: the unit of target outputs in different regions are not totally same

        Parameters
        ----------
        gage_id_lst
            station ids
        t_range
            the time range, for example, ["1990-01-01", "2000-01-01"]
        target_cols
            the default is None, but we need at least one default target.
            For CAMELS-AUS, it's ["streamflow_mmd"]
        kwargs
            some other params if needed

        Returns
        -------
        np.array
            streamflow data, 3-dim [station, time, streamflow]
        """
        if target_cols is None:
            return np.array([])
        else:
            nf = len(target_cols)
        t_range_list = hydro_time.t_range_days(t_range)
        nt = t_range_list.shape[0]
        y = np.full([len(gage_id_lst), nt, nf], np.nan)
        for k in tqdm(
            range(len(target_cols)), desc="Read streamflow data of CAMELS-AUS"
        ):
            flow_data = pd.read_csv(
                os.path.join(
                    self.data_source_description["CAMELS_FLOW_DIR"],
                    target_cols[k] + ".csv",
                )
            )
            df_date = flow_data[["year", "month", "day"]]
            date = pd.to_datetime(df_date).values.astype("datetime64[D]")
            [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
            chosen_data = flow_data[gage_id_lst].values[ind1, :]
            chosen_data[chosen_data < 0] = np.nan
            y[:, ind2, k] = chosen_data.T
            # ML/d-->m3/s
            if target_cols[k] == "streamflow_MLd":
                y = y / 84.6
        # Keep unit of streamflow unified: we use ft3/s here
        # other units are m3/s -> ft3/s
        y = y * 35.314666721489
        return y

    def read_relevant_cols(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        forcing_type="daymet",
        **kwargs,
    ) -> np.ndarray:
        """
        Read forcing data

        Parameters
        ----------
        gage_id_lst
            station ids
        t_range
            the time range, for example, ["1990-01-01", "2000-01-01"]
        var_lst
            forcing variable types
        forcing_type
            now only for CAMELS-US, there are three types: daymet, nldas, maurer
        Returns
        -------
        np.array
            forcing data
        """
        t_range_list = hydro_time.t_range_days(t_range)
        nt = t_range_list.shape[0]
        x = np.full([len(gage_id_lst), nt, len(var_lst)], np.nan)
        for k in tqdm(range(len(var_lst)), desc="Read forcing data of CAMELS-AUS"):
            if "precipitation_" in var_lst[k]:
                forcing_dir = os.path.join(
                    self.data_source_description["CAMELS_FORCING_DIR"],
                    "01_precipitation_timeseries",
                )
            elif "et_" in var_lst[k] or "evap_" in var_lst[k]:
                forcing_dir = os.path.join(
                    self.data_source_description["CAMELS_FORCING_DIR"],
                    "02_EvaporativeDemand_timeseries",
                )
            elif "_AWAP" in var_lst[k]:
                forcing_dir = os.path.join(
                    self.data_source_description["CAMELS_FORCING_DIR"],
                    "03_Other",
                    "AWAP",
                )
            elif "_SILO" in var_lst[k]:
                forcing_dir = os.path.join(
                    self.data_source_description["CAMELS_FORCING_DIR"],
                    "03_Other",
                    "SILO",
                )
            else:
                raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)
            forcing_data = pd.read_csv(os.path.join(forcing_dir, var_lst[k] + ".csv"))
            df_date = forcing_data[["year", "month", "day"]]
            date = pd.to_datetime(df_date).values.astype("datetime64[D]")
            [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
            chosen_data = forcing_data[gage_id_lst].values[ind1, :]
            x[:, ind2, k] = chosen_data.T
        return x

    def read_attr_all_in_one_file(self):
        """
        Read all attr data in CAMELS_AUS

        Returns
        -------
        np.array
            all attr data in CAMELS_AUS
        """
        attr_all_file = os.path.join(
            self.data_source_description["CAMELS_DIR"],
            "CAMELS_AUS_Attributes-Indices_MasterTable.csv",
        )
        all_attr = pd.read_csv(attr_all_file, sep=",")
        var_lst = self.get_constant_cols().tolist()
        data_temp = all_attr[var_lst]
        # for factorized data, we need factorize all gages' data to keep the factorized number same all the time
        n_gage = len(self.read_object_ids())
        out = np.full([n_gage, len(var_lst)], np.nan)
        f_dict = {}
        k = 0
        for field in var_lst:
            if is_string_dtype(data_temp[field]):
                value, ref = pd.factorize(data_temp[field], sort=True)
                out[:, k] = value
                f_dict[field] = ref.tolist()
            elif is_numeric_dtype(data_temp[field]):
                out[:, k] = data_temp[field].values
            k = k + 1
        # keep same format with CAMELS_US
        return out, var_lst, None, f_dict

    def read_constant_cols(
        self, gage_id_lst=None, var_lst=None, is_return_dict=False, **kwargs
    ) -> np.ndarray:
        """
        Read Attributes data

        Parameters
        ----------
        gage_id_lst
            station ids
        var_lst
            attribute variable types
        is_return_dict
            if true, return var_dict and f_dict for CAMELS_US
        Returns
        -------
        Union[tuple, np.array]
            if attr var type is str, return factorized data.
            When we need to know what a factorized value represents, we need return a tuple;
            otherwise just return an array
        """
        attr_all, var_lst_all, var_dict, f_dict = self.read_attr_all_in_one_file()
        ind_var = [var_lst_all.index(var) for var in var_lst]
        id_lst_all = self.read_object_ids()
        # Notice the sequence of station ids ! Some id_lst_all are not sorted, so don't use np.intersect1d
        ind_grid = [id_lst_all.tolist().index(tmp) for tmp in gage_id_lst]
        temp = attr_all[ind_grid, :]
        out = temp[:, ind_var]
        return (out, var_dict, f_dict) if is_return_dict else out

    def read_area(self, gage_id_lst) -> np.ndarray:
        return self.read_constant_cols(
            gage_id_lst, ["catchment_area"], is_return_dict=False
        )

    def read_mean_prcp(self, gage_id_lst, unit="mm/d") -> xr.Dataset:
        """Read mean precipitation data

        Parameters
        ----------
        gage_id_lst : list
            station ids
        unit : str, optional
            the unit of mean_prcp, by default "mm/d"

        Returns
        -------
        xr.Dataset
            mean precipitation data
        """
        data = self.read_constant_cols(
            gage_id_lst,
            ["p_mean"],
            is_return_dict=False,
        )
        if unit in ["mm/d", "mm/day"]:
            converted_data = data
        elif unit in ["mm/h", "mm/hour"]:
            converted_data = data / 24
        elif unit in ["mm/3h", "mm/3hour"]:
            converted_data = data / 8
        elif unit in ["mm/8d", "mm/8day"]:
            converted_data = data * 8
        else:
            raise ValueError(
                "unit must be one of ['mm/d', 'mm/day', 'mm/h', 'mm/hour', 'mm/3h', 'mm/3hour', 'mm/8d', 'mm/8day']"
            )
        return converted_data
