import fnmatch
import os
import collections
import pandas as pd
import numpy as np
from pathlib import Path
from pandas.api.types import is_string_dtype, is_numeric_dtype
from typing import Union
from tqdm import tqdm
from hydroutils import hydro_time, hydro_file
from hydrodataset import CACHE_DIR, HydroDataset, CAMELS_REGIONS
from hydrodataset.camels import Camels

CAMELS_NO_DATASET_ERROR_LOG = (
    "We cannot read this dataset now. Please check if you choose correctly:\n"
    + str(CAMELS_REGIONS)
)


class CamelsCl(Camels):
    def __init__(
        self,
        data_path=os.path.join("camels", "camels_cl"),
        download=False,
        region: str = "CL",
    ):
        """
        Initialization for CAMELS-CL dataset

        Parameters
        ----------
        data_path
            where we put the dataset.
            we already set the ROOT directory for hydrodataset,
            so here just set it as a relative path,
            by default "camels/camels_cl"
        download
            if true, download, by default False
        region
            the default is CAMELS-CL
        """
        super().__init__(data_path, download, region)

    def set_data_source_describe(self) -> collections.OrderedDict:
        """
        the files in the dataset and their location in file system

        Returns
        -------
        collections.OrderedDict
            the description for a CAMELS-CL dataset
        """
        if self.region != "CL":
            raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)
        camels_db = self.data_source_dir
        return self._set_data_source_camelscl_describe(camels_db)

    def _set_data_source_camelscl_describe(self, camels_db):
        # attr
        attr_dir = camels_db.joinpath("1_CAMELScl_attributes")
        attr_file = attr_dir.joinpath("1_CAMELScl_attributes.txt")
        # shp file of basins
        camels_shp_file = camels_db.joinpath(
            "CAMELScl_catchment_boundaries",
            "catchments_camels_cl_v1.3.shp",
        )
        # config of flow data
        flow_dir_m3s = camels_db.joinpath("2_CAMELScl_streamflow_m3s")
        flow_dir_mm = camels_db.joinpath("3_CAMELScl_streamflow_mm")

        # forcing
        forcing_dir_precip_cr2met = camels_db.joinpath("4_CAMELScl_precip_cr2met")
        forcing_dir_precip_chirps = camels_db.joinpath("5_CAMELScl_precip_chirps")
        forcing_dir_precip_mswep = camels_db.joinpath("6_CAMELScl_precip_mswep")
        forcing_dir_precip_tmpa = camels_db.joinpath("7_CAMELScl_precip_tmpa")
        forcing_dir_tmin_cr2met = camels_db.joinpath("8_CAMELScl_tmin_cr2met")
        forcing_dir_tmax_cr2met = camels_db.joinpath("9_CAMELScl_tmax_cr2met")
        forcing_dir_tmean_cr2met = camels_db.joinpath("10_CAMELScl_tmean_cr2met")
        forcing_dir_pet_8d_modis = camels_db.joinpath("11_CAMELScl_pet_8d_modis")
        forcing_dir_pet_hargreaves = camels_db.joinpath("12_CAMELScl_pet_hargreaves")
        forcing_dir_swe = camels_db.joinpath("13_CAMELScl_swe")
        return collections.OrderedDict(
            CAMELS_DIR=camels_db,
            CAMELS_FLOW_DIR=[flow_dir_m3s, flow_dir_mm],
            CAMELS_FORCING_DIR=[
                forcing_dir_precip_cr2met,
                forcing_dir_precip_chirps,
                forcing_dir_precip_mswep,
                forcing_dir_precip_tmpa,
                forcing_dir_tmin_cr2met,
                forcing_dir_tmax_cr2met,
                forcing_dir_tmean_cr2met,
                forcing_dir_pet_8d_modis,
                forcing_dir_pet_hargreaves,
                forcing_dir_swe,
            ],
            CAMELS_ATTR_DIR=attr_dir,
            CAMELS_GAUGE_FILE=attr_file,
            CAMELS_BASINS_SHP_FILE=camels_shp_file,
        )

    def read_site_info(self) -> pd.DataFrame:
        """
        Read the basic information of gages in a CAMELS-CL dataset

        Returns
        -------
        pd.DataFrame
            basic info of gages
        """
        camels_file = self.data_source_description["CAMELS_GAUGE_FILE"]
        return pd.read_csv(camels_file, sep="\t", index_col=0)

    def get_constant_cols(self) -> np.ndarray:
        """
        all readable attrs in CAMELS-CL

        Returns
        -------
        np.array
            attribute types
        """
        camels_cl_attr_data = self.sites
        # exclude station id
        return camels_cl_attr_data.index.values

    def get_relevant_cols(self) -> np.ndarray:
        """
        all readable forcing types in CAMELS-CL

        Returns
        -------
        np.array
            forcing types
        """
        return np.array(
            [
                "_".join(str(forcing_dir).split(os.sep)[-1].split("_")[2:])
                for forcing_dir in self.data_source_description["CAMELS_FORCING_DIR"]
            ]
        )

    def get_target_cols(self) -> np.ndarray:
        """
        For CAMELS-CL, the target vars are streamflows

        Returns
        -------
        np.array
            streamflow types
        """
        return np.array(
            [
                str(flow_dir).split(os.sep)[-1][11:]
                for flow_dir in self.data_source_description["CAMELS_FLOW_DIR"]
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
        station_ids = self.sites.columns.values
        # for 7-digit id, replace the space with 0 to get a 8-digit id
        cl_station_ids = [
            station_id.split(" ")[-1].zfill(8) for station_id in station_ids
        ]
        return np.array(cl_station_ids)

    def read_target_cols(
        self,
        gage_id_lst: Union[list, np.array] = None,
        t_range: list = None,
        target_cols: Union[list, np.array] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        read target values; for CAMELS-CL, they are streamflows

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
            For CAMELS-CL, it's ["streamflow_m3s"]
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
            range(len(target_cols)), desc="Read streamflow data of CAMELS-CL"
        ):
            if target_cols[k] == "streamflow_m3s":
                flow_data = pd.read_csv(
                    os.path.join(
                        self.data_source_description["CAMELS_FLOW_DIR"][0],
                        "2_CAMELScl_streamflow_m3s.txt",
                    ),
                    sep="\t",
                    index_col=0,
                )
            elif target_cols[k] == "streamflow_mm":
                flow_data = pd.read_csv(
                    os.path.join(
                        self.data_source_description["CAMELS_FLOW_DIR"][1],
                        "3_CAMELScl_streamflow_mm.txt",
                    ),
                    sep="\t",
                    index_col=0,
                )
            else:
                raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)
            date = pd.to_datetime(flow_data.index.values).values.astype("datetime64[D]")
            [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
            station_ids = [id_.zfill(8) for id_ in flow_data.columns.values]
            assert all(x < y for x, y in zip(station_ids, station_ids[1:]))
            ind3 = [station_ids.index(tmp) for tmp in gage_id_lst]
            # to guarantee the sequence is not changed we don't use np.intersect1d
            chosen_data = flow_data.iloc[ind1, ind3].replace("\s+", np.nan, regex=True)
            chosen_data = chosen_data.astype(float)
            chosen_data[chosen_data < 0] = np.nan
            y[:, ind2, k] = chosen_data.values.T
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
        for k in tqdm(range(len(var_lst)), desc="Read forcing data of CAMELS-CL"):
            for tmp in os.listdir(self.data_source_description["CAMELS_DIR"]):
                if fnmatch.fnmatch(tmp, "*" + var_lst[k]):
                    tmp_ = os.path.join(self.data_source_description["CAMELS_DIR"], tmp)
                    if os.path.isdir(tmp_):
                        forcing_file = os.path.join(tmp_, os.listdir(tmp_)[0])
            forcing_data = pd.read_csv(forcing_file, sep="\t", index_col=0)
            date = pd.to_datetime(forcing_data.index.values).values.astype(
                "datetime64[D]"
            )
            [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
            station_ids = [id_.zfill(8) for id_ in forcing_data.columns.values]
            assert all(x < y for x, y in zip(station_ids, station_ids[1:]))
            ind3 = [station_ids.index(tmp) for tmp in gage_id_lst]
            # to guarantee the sequence is not changed we don't use np.intersect1d
            chosen_data = forcing_data.iloc[ind1, ind3].replace(
                "\s+", np.nan, regex=True
            )
            x[:, ind2, k] = chosen_data.values.T
        return x

    def read_attr_all_in_one_file(self):
        """
        Read all attr data in CAMELS_CL

        Returns
        -------
        np.array
            all attr data in CAMELS_CL
        """
        attr_all_file = os.path.join(
            self.data_source_description["CAMELS_ATTR_DIR"],
            "1_CAMELScl_attributes.txt",
        )
        all_attr_tmp = pd.read_csv(attr_all_file, sep="\t", index_col=0)
        all_attr = pd.DataFrame(
            all_attr_tmp.values.T,
            index=all_attr_tmp.columns,
            columns=all_attr_tmp.index,
        )
        # some none str attributes are treated as str, we need to trans them to float
        all_cols = all_attr.columns
        for col in all_cols:
            try:
                all_attr[col] = all_attr[col].astype(float)
            except Exception:
                continue
        # gage_all_attr = all_attr[all_attr['station_id'].isin(gage_id_lst)]
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
        return self.read_constant_cols(gage_id_lst, ["area"], is_return_dict=False)

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
        # there are different p_mean values for different forcings, here we chose p_mean_cr2met now
        data = self.read_constant_cols(
            gage_id_lst, ["p_mean_cr2met"], is_return_dict=False
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
