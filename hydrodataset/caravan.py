import logging
import collections
import glob
import re
import warnings
from tqdm import tqdm
import xarray as xr
import shutil
import os
from pathlib import Path
from typing import Union
import tarfile
from urllib.request import urlopen
import pandas as pd
import numpy as np
from tzfpy import get_tz
from hydroutils import hydro_file
from hydrodataset import CACHE_DIR, HydroDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
class Caravan(HydroDataset):
    def __init__(self, data_path, download=False, region="Global"):
        """
        Initialization for Caravan dataset

        Parameters
        ----------
        data_path
            where we put the dataset
        download
            if true, download
        region
            the region can be US, AUS, BR, CL, GB, CE, NA (North America, meaning HYSETS)
        """
        super().__init__(data_path)
        self.region = region
        region_name_dict = self.region_name_dict
        if region == "Global":
            self.region_data_name = list(region_name_dict.values())
        else:
            self.region_data_name = region_name_dict[region]
        self.data_source_description = self.set_data_source_describe()
        if download:
            self.download_data_source()
        try:
            self.is_data_ready()
        except FileNotFoundError as e:
            warnings.warn(e)
            print(
                "Please download and unzip the dataset first: just set download=True if you have manually downloaded zip files when you first initialize caravan."
            )
        self.sites = self.read_site_info()

    @property
    def region_name_dict(self):
        return {
            "US": "camels",
            "AUS": "camelsaus",
            "BR": "camelsbr",
            "CL": "camelscl",
            "GB": "camelsgb",
            "NA": "hysets",
            "CE": "lamah",
        }

    def get_name(self):
        return "Caravan_" + self.region

    def set_data_source_describe(self) -> collections.OrderedDict:
        """
        Introduce the files in the dataset and list their location in the file system

        Returns
        -------
        collections.OrderedDict
            the description for Caravan
        """
        dataset_dir = self._base_dir()

        # We use A_basins_total_upstrm
        # shp file of basins
        # TODO: Caravan Global is not fully tested yet
        shp_dir = os.path.join(dataset_dir, "shapefiles", self.region_data_name)
        # read the shp in this dir
        shp_files = [f for f in os.listdir(shp_dir) if f.endswith(".shp")]

        if len(shp_files) != 1:
            raise ValueError(
                f"Expected one shapefile in {shp_dir}, found {len(shp_files)}"
            )
        shp_file_path = os.path.join(shp_dir, shp_files[0])
        # config of flow data
        flow_dir = os.path.join(dataset_dir, "timeseries", "netcdf")
        forcing_dir = flow_dir
        attr_dir = os.path.join(dataset_dir, "attributes")
        ts_csv_dir = os.path.join(dataset_dir, "timeseries", "csv")
        download_url = "https://zenodo.org/records/14673536/files/Caravan-nc.tar.xz?download=1"
        return collections.OrderedDict(
            DATASET_DIR=dataset_dir,
            FLOW_DIR=flow_dir,
            FORCING_DIR=forcing_dir,
            TS_CSV_DIR=ts_csv_dir,
            ATTR_DIR=attr_dir,
            BASINS_SHP_FILE=shp_file_path,
            DOWNLOAD_URL=download_url,
        )

    def _base_dir(self):
        return os.path.join(self.data_source_dir, "Caravan", "Caravan")

    def download_data_source(self) -> None:
        """
        Download dataset.

        Returns
        -------
        None
        """
        dataset_config = self.data_source_description
        self.data_source_dir.mkdir(exist_ok=True)
        url = dataset_config["DOWNLOAD_URL"]
        fzip = Path(self.data_source_dir, url.rsplit("/", 1)[1])
        print(f"Checking file: {fzip}")
        if fzip.exists():
            with urlopen(url) as response:
                if int(response.info()["Content-length"]) != fzip.stat().st_size:
                    fzip.unlink()  # Remove the incomplete file if the size does not match
        
        # Prepare the download list if the file doesn't exist
        to_dl = []
        if not fzip.exists():
            to_dl.append(url)
        
        # Download the file if needed
        if to_dl:
            hydro_file.download_zip_files(to_dl, self.data_source_dir)
        
        # Handle unzipping .tar.xz files
        try:
            # If the downloaded file is a .tar.xz file, extract it
            if fzip.suffix == '.xz':
                with tarfile.open(fzip, "r:xz") as tar:
                    tar.extractall(path=self.data_source_dir)
                logging.info(f"Successfully extracted {fzip.name}")
            elif fzip.suffix == '.zip':
                # Handle regular .zip files (just in case)
                shutil.unpack_archive(fzip, self.data_source_dir)
                logging.info(f"Successfully extracted {fzip.name}")
            else:
                logging.warning(f"Unsupported file format: {fzip.suffix}")
            
        except tarfile.ReadError:
            Warning("Please manually unzip the file.")

    def is_data_ready(self):
        """Check if the data is ready to be read"""
        if not os.path.exists(self.data_source_description["DATASET_DIR"]):
            raise FileNotFoundError(
                f"Dataset is not found in {self.data_source_description['DATASET_DIR']}"
            )
        if not os.path.exists(self.data_source_description["FLOW_DIR"]):
            raise FileNotFoundError(
                f"Flow data is not found in {self.data_source_description['FLOW_DIR']}"
            )
        if not os.path.exists(self.data_source_description["FORCING_DIR"]):
            raise FileNotFoundError(
                f"Forcing data is not found in {self.data_source_description['FORCING_DIR']}"
            )
        if not os.path.exists(self.data_source_description["ATTR_DIR"]):
            raise FileNotFoundError(
                f"Attributes data is not found in {self.data_source_description['ATTR_DIR']}"
            )

    def read_site_info(self) -> pd.DataFrame:
        """
        Read the basic information of gages in dataset

        Returns
        -------
        pd.DataFrame
            basic info of gages
        """
        if self.region == "Global":
            attr = []
            for region in self.region_data_name:
                site_file = os.path.join(
                    self.data_source_description["ATTR_DIR"],
                    region,
                    "attributes_caravan_" + region + ".csv",
                )
                attr_region = pd.read_csv(site_file, sep=",")
                attr.append(attr_region)
            return pd.concat(attr)
        site_file = os.path.join(
            self.data_source_description["ATTR_DIR"],
            self.region_data_name,
            "attributes_caravan_" + self.region_data_name + ".csv",
        )
        return pd.read_csv(site_file, sep=",")

    def get_constant_cols(self) -> np.array:
        """
        all readable attrs

        Returns
        -------
        np.array
            attribute types
        """
        if self.region == "Global":
            attr_types = []
            for region in self.region_data_name:
                attr_indices = self._attr_columns_region(region)
                attr_types.append(attr_indices)
            return np.unique(np.concatenate(attr_types))
        return self._attr_columns_region(self.region_data_name)

    def _attr_columns_region(self, region):
        attr_file1 = os.path.join(
            self.data_source_description["DATASET_DIR"],
            "attributes",
            region,
            "attributes_caravan_" + region + ".csv",
        )
        attr_indices_data1 = pd.read_csv(attr_file1, sep=",")
        attr_file2 = os.path.join(
            self.data_source_description["DATASET_DIR"],
            "attributes",
            region,
            "attributes_hydroatlas_" + region + ".csv",
        )
        attr_indices_data2 = pd.read_csv(attr_file2, sep=",")
        attr_file3 = os.path.join(
            self.data_source_description["DATASET_DIR"],
            "attributes",
            region,
            "attributes_other_" + region + ".csv",
        )
        attr_indices_data3 = pd.read_csv(attr_file3, sep=",")
        return np.array(
            attr_indices_data1.columns.values[1:].tolist()
            + attr_indices_data2.columns.values[1:].tolist()
            + attr_indices_data3.columns.values[1:].tolist()
        )

    def get_relevant_cols(self) -> np.array:
        """
        all readable forcing types, also including streamflow

        Returns
        -------
        np.array
            forcing types
        """
        if self.region == "Global":
            forcing_types = []
            for region in self.region_data_name:
                forcing_dir = os.path.join(
                    self.data_source_description["FORCING_DIR"],
                    region,
                )
                if not (files := os.listdir(forcing_dir)):
                    raise FileNotFoundError("No files found in the directory.")
                first_file = files[0]
                file_path = os.path.join(forcing_dir, first_file)
                data = xr.open_dataset(file_path)
                forcing_types.append(list(data.data_vars))
            return np.unique(np.concatenate(forcing_types))
        forcing_dir = os.path.join(
            self.data_source_description["FORCING_DIR"],
            self.region_data_name,
        )
        if not (files := os.listdir(forcing_dir)):
            raise FileNotFoundError("No files found in the directory.")
        first_file = files[0]
        file_path = os.path.join(forcing_dir, first_file)

        if files := os.listdir(forcing_dir):
            first_file = files[0]
            file_path = os.path.join(forcing_dir, first_file)
            data = xr.open_dataset(file_path)
        else:
            raise FileNotFoundError("No files found in the directory.")
        return np.array(list(data.data_vars))

    def get_target_cols(self) -> np.array:
        """
        the target vars are streamflows

        Returns
        -------
        np.array
            streamflow types
        """
        return np.array(["streamflow"])

    def read_object_ids(self, **kwargs) -> np.array:
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
        return self.sites["gauge_id"].values

    def read_target_cols(
        self,
        gage_id_lst: Union[list, np.array] = None,
        t_range: list = None,
        target_cols: Union[list, np.array] = None,
        **kwargs,
    ) -> np.array:
        if self.region == "Global":
            return self._read_timeseries_data_global(
                "FLOW_DIR", gage_id_lst, t_range, target_cols
            )
        return self._read_timeseries_data("FLOW_DIR", gage_id_lst, t_range, target_cols)

    def _read_timeseries_data_global(self, dir_name, gage_id_lst, t_range, var_lst):
        ts_dir = self.data_source_description[dir_name]
        if gage_id_lst is None:
            gage_id_lst = self.read_object_ids()
        # Find matching file paths
        file_paths = []
        for region in self.region_data_name:
            for file_name in gage_id_lst:
                file_path = os.path.join(ts_dir, region, file_name) + ".nc"
                if os.path.isfile(file_path):
                    file_paths.append(file_path)
        datasets = [
            xr.open_dataset(path).assign_coords(gauge_id=name)
            for path, name in zip(file_paths, gage_id_lst)
        ]
        # Concatenate the datasets along the new dimension
        data = xr.concat(datasets, dim="gauge_id").sortby("gauge_id")
        if t_range is not None:
            data = data.sel(date=slice(t_range[0], t_range[1]))
        if var_lst is None:
            if dir_name == "FLOW_DIR":
                var_lst = self.get_target_cols()
            else:
                var_lst = self.get_relevant_cols()
        return data[var_lst]

    def read_relevant_cols(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        forcing_type="era5land",
        **kwargs,
    ) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        gage_id_lst : list, optional
            _description_, by default None
        t_range : list, optional
            A special notice is that for xarray, the time range is [start, end] which is a closed interval.
        var_lst : list, optional
            _description_, by default None
        forcing_type : str, optional
            _description_, by default "daymet"

        Returns
        -------
        np.ndarray
            _description_
        """
        if self.region == "Global":
            return self._read_timeseries_data_global(
                "FORCING_DIR", gage_id_lst, t_range, var_lst
            )
        return self._read_timeseries_data("FORCING_DIR", gage_id_lst, t_range, var_lst)

    def _read_timeseries_data(self, dir_name, gage_id_lst, t_range, var_lst):
        ts_dir = self.data_source_description[dir_name]
        if gage_id_lst is None:
            gage_id_lst = self.read_object_ids()
        # Find matching file paths
        file_paths = []
        for file_name in gage_id_lst:
            file_path = os.path.join(ts_dir, self.region_data_name, file_name) + ".nc"
            if os.path.isfile(file_path):
                file_paths.append(file_path)
        datasets = [
            xr.open_dataset(path).assign_coords(gauge_id=name)
            for path, name in zip(file_paths, gage_id_lst)
        ]
        # Concatenate the datasets along the new dimension
        data = xr.concat(datasets, dim="gauge_id").sortby("gauge_id")
        if t_range is not None:
            data = data.sel(date=slice(t_range[0], t_range[1]))
        if var_lst is None:
            if dir_name == "FLOW_DIR":
                var_lst = self.get_target_cols()
            else:
                var_lst = self.get_relevant_cols()
        return data[var_lst]

    def read_constant_cols(
        self, gage_id_lst=None, var_lst=None, is_return_dict=False, **kwargs
    ):
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
            When we need to know what a factorized value represents,
            we need return a tuple;
            otherwise just return an array
        """
        if self.region == "Global":
            return self._read_constant_cols_global(var_lst, gage_id_lst, is_return_dict)
        data = self._read_attr_files_1region(
            self.region_data_name, gage_id_lst, var_lst
        )
        return data.to_dict("index") if is_return_dict else data.values

    def _read_attr_files_1region(self, region, gage_id_lst, var_lst):
        """When gage_id_lst is None, we read all gages in this region;
        when var_lst is None, we read all attributes in this region

        Parameters
        ----------
        region : str
            region name
        gage_id_lst : list
            gage ids
        var_lst : list
            attribute variable types

        Returns
        -------
        pd.DataFrame
            attributes data
        """
        attr_file1 = os.path.join(
            self.data_source_description["DATASET_DIR"],
            "attributes",
            region,
            "attributes_caravan_" + region + ".csv",
        )
        attr_file2 = os.path.join(
            self.data_source_description["DATASET_DIR"],
            "attributes",
            region,
            "attributes_hydroatlas_" + region + ".csv",
        )
        attr_file3 = os.path.join(
            self.data_source_description["DATASET_DIR"],
            "attributes",
            region,
            "attributes_other_" + region + ".csv",
        )
        data1 = pd.read_csv(attr_file1, sep=",", dtype={"gauge_id": str})
        data1 = data1.set_index("gauge_id")
        data2 = pd.read_csv(attr_file2, sep=",", dtype={"gauge_id": str})
        data2 = data2.set_index("gauge_id")
        data3 = pd.read_csv(attr_file3, sep=",", dtype={"gauge_id": str})
        data3 = data3.set_index("gauge_id")
        data = pd.concat([data1, data2, data3], axis=1)
        if gage_id_lst is not None:
            data = data.loc[gage_id_lst]
        if var_lst is not None:
            data = data.loc[:, var_lst]
        return data

    def _read_constant_cols_global(self, var_lst, gage_id_lst, is_return_dict):
        attr = []
        if var_lst is None:
            var_lst = self.get_constant_cols()
        if gage_id_lst is None:
            gage_id_lst = self.read_object_ids()
        for region in self.region_data_name:
            # as the gage_id may come from different regions, to avoid error, we set gage_id_lst=None
            attr_indices_data = self._read_attr_files_1region(
                region, gage_id_lst=None, var_lst=var_lst
            )
            gage_in_this_region = np.intersect1d(
                attr_indices_data.index.values, gage_id_lst
            )
            if gage_in_this_region.size > 0:
                attr_indices_data = attr_indices_data.loc[gage_in_this_region]
                attr_indices_data = attr_indices_data.loc[:, var_lst]
                attr.append(attr_indices_data)
        data = pd.concat(attr)
        return data.to_dict("index") if is_return_dict else data.values

    def read_mean_prep(self, object_ids) -> np.array:
        return self.read_constant_cols(object_ids, ["p_mean"], is_return_dict=False)

    def cache_attributes_xrdataset(self):
        """cache attributes in xr dataset"""
        import pint_xarray  # noqa
        import pint
        from pint import UnitRegistry

        data = self.read_constant_cols()
        basin_ids = self.read_object_ids()
        var_names = self.get_constant_cols()
        assert all(x <= y for x, y in zip(basin_ids, basin_ids[1:]))
        if self.region == "Global":
            # for all attrs reading in Global mode, all attrs are sorted
            assert all(x <= y for x, y in zip(var_names, var_names[1:]))
        ds = xr.Dataset(
            {var: (["basin"], data[:, i]) for i, var in enumerate(var_names)},
            coords={"basin": basin_ids},
        )
        units_dict = {attribute: "" for attribute in var_names}
        # Define a dictionary for the provided attribute patterns and their units
        # https://data.hydrosheds.org/file/technical-documentation/BasinATLAS_Catalog_v10.pdf
        units_mapping = {
            "dis_m3_": "cubic meters/second",
            "run_mm_": "millimeters",
            "inu_pc_": "percent cover",
            "lka_pc_": "percent cover (x10)",
            "lkv_mc_": "million cubic meters",
            "rev_mc_": "million cubic meters",
            "dor_pc_": "percent (x10)",
            "ria_ha_": "hectares",
            "riv_tc_": "thousand cubic meters",
            "gwt_cm_": "centimeters",
            "ele_mt_": "meters a.s.l.",
            "slp_dg_": "degrees (x10)",
            "sgr_dk_": "decimeters per km",
            "clz_cl_": "classes (18)",
            "cls_cl_": "classes (125)",
            "tmp_dc_": "degrees Celsius (x10)",
            "pre_mm_": "millimeters",
            "pet_mm_": "millimeters",
            "aet_mm_": "millimeters",
            "ari_ix_": "index value (x100)",
            "cmi_ix_": "index value (x100)",
            "snw_pc_": "percent cover",
            "glc_cl_": "classes (22)",
            "glc_pc_": "percent cover",
            "pnv_cl_": "classes (15)",
            "pnv_pc_": "percent cover",
            "wet_cl_": "classes (12)",
            "wet_pc_": "percent cover",
            "for_pc_": "percent cover",
            "crp_pc_": "percent cover",
            "pst_pc_": "percent cover",
            "ire_pc_": "percent cover",
            "gla_pc_": "percent cover",
            "prm_pc_": "percent cover",
            "pac_pc_": "percent cover",
            "tbi_cl_": "classes (14)",
            "tec_cl_": "classes (846)",
            "fmh_cl_": "classes (13)",
            "fec_cl_": "classes (426)",
            "cly_pc_": "percent",
            "slt_pc_": "percent",
            "snd_pc_": "percent",
            "soc_th_": "tonnes/hectare",
            "swc_pc_": "percent",
            "lit_cl_": "classes (16)",
            "kar_pc_": "percent cover",
            "ero_kh_": "kg/hectare per year",
            "pop_ct_": "count (thousands)",
            "ppd_pk_": "people per km²",
            "urb_pc_": "percent cover",
            "nli_ix_": "index value (x100)",
            "rdd_mk_": "meters per km²",
            "hft_ix_": "index value (x10)",
            "gad_id_": "ID number",
            "gdp_ud_": "US dollars",
            "hdi_ix_": "index value (x1000)",
        }

        # Update the attributes_dict based on the units_mapping
        for key_pattern, unit in units_mapping.items():
            for key in units_dict:
                if key.startswith(key_pattern):
                    units_dict[key] = unit

        # for attrs not from hydroatlas in caravan, we directly set pint unit
        units_dict["area"] = "km^2"
        units_dict["area_fraction_used_for_aggregation"] = (
            "dimensionless"  # this one is from atlas but not specified in the document
        )
        units_dict["aridity"] = "dimensionless"
        units_dict["country"] = "dimensionless"
        units_dict["frac_snow"] = "dimensionless"
        units_dict["gauge_lat"] = "degree"
        units_dict["gauge_lon"] = "degree"
        units_dict["gauge_name"] = "dimensionless"
        units_dict["high_prec_dur"] = "day"
        units_dict["high_prec_freq"] = "day/year"
        units_dict["low_prec_dur"] = "day"
        units_dict["low_prec_freq"] = "day/year"
        units_dict["moisture_index"] = "dimensionless"
        units_dict["p_mean"] = "mm/year"
        units_dict["pet_mean"] = "mm/year"
        units_dict["seasonality"] = "dimensionless"

        # Reinitialize unit registry and unit mapping dictionary
        ureg = UnitRegistry()

        pint_unit_mapping = {
            "cubic meters/second": "m^3/s",
            "millimeters": "millimeter",
            "percent cover": "percent",
            "percent cover (x10)": "1e-1 * percent",
            "million cubic meters": "1e6 * m^3",
            "thousand cubic meters": "1e3 * m^3",
            "centimeters": "centimeter",
            "meters a.s.l.": "meter",
            "degrees (x10)": "1e-1 * degree",
            "decimeters per km": "decimeter/km",
            "classes (18)": "dimensionless",
            "classes (125)": "dimensionless",
            "degrees Celsius (x10)": "degree_Celsius",
            "index value (x100)": "1e-2",
            "classes (22)": "dimensionless",
            "classes (15)": "dimensionless",
            "classes (12)": "dimensionless",
            "classes (14)": "dimensionless",
            "classes (846)": "dimensionless",
            "classes (13)": "dimensionless",
            "classes (426)": "dimensionless",
            "percent": "percent",
            "tonnes/hectare": "tonne/hectare",
            "kg/hectare per year": "kg/hectare/year",
            "count (thousands)": "1e3",
            "people per km²": "1/km^2",
            "index value (x1000)": "1e-3",
            "meters per km²": "meter/km^2",
            "index value (x10)": "1e-1",
            "US dollars": "dimensionless",
            "ID number": "dimensionless",
        }

        # Validate each unit in the unit_mapping dictionary
        valid_units = {}
        invalid_units = {}

        for provided_unit, pint_unit in pint_unit_mapping.items():
            try:
                ureg.parse_expression(pint_unit)
                valid_units[provided_unit] = pint_unit
            except pint.errors.UndefinedUnitError:
                invalid_units[provided_unit] = pint_unit

        converted_units = {
            var: pint_unit_mapping.get(unit, unit) for var, unit in units_dict.items()
        }
        assert list(units_dict.keys()) == list(
            converted_units.keys()
        ), "The keys of the dictionaries don't match or are not in the same order!"

        # for tmp_dc_ variable, we can't convert its unit to 0.1 * degree_Celsius
        # hence we turn its value to degree_Celsius
        for var in ds.data_vars:
            if var.startswith("tmp_dc_"):
                ds[var] = ds[var] * 0.1

        # Assign units to the variables in the Dataset
        for var_name in converted_units:
            if var_name in ds.data_vars:
                ds[var_name].attrs["units"] = converted_units[var_name]
        region_name = self.region_data_name
        if isinstance(region_name, list):
            region_name = "_".join(region_name)
        cache_attr_file = os.path.join(
            CACHE_DIR,
            f"caravan_{region_name}_attributes.nc",
        )
        ds.to_netcdf(cache_attr_file)

    def cache_xrdataset(self, **kwargs):
        """
        Save all attr data in a netcdf file in the cache directory,
        ts data are already nc format

        Parameters
        ----------
        checkregion : str, optional
            as my experience, the dameged file is in hysets, by default "hysets"
        """
        checkregion = kwargs.get("checkregion", "hysets")
        warnings.warn("Check you units of all variables")
        self.cache_attributes_xrdataset()
        self.cache_timeseries_xrdataset(checkregion)

    def read_timeseries(self, basin_ids=None, t_range_list: list = None, var_lst=None):
        """
        Read time-series data from csv files

        Parameters
        ----------
        basin_ids
            station ids
        t_range
            time range
        var_lst
            relevant columns

        Returns
        -------
        np.array
            time-series data
        """
        if basin_ids is None:
            basin_ids = self.read_object_ids()
        if var_lst is None:
            var_lst = self.get_relevant_cols()
        if t_range_list is None:
            t_range_list = ["1980-01-01", "2023-12-31"]
        ts_dir = self.data_source_description["TS_CSV_DIR"]
        region = self.region_data_name
        t_range = pd.date_range(start=t_range_list[0], end=t_range_list[-1], freq="1D")
        nt = len(t_range)
        x = np.full([len(basin_ids), nt, len(var_lst)], np.nan)

        for k in tqdm(range(len(basin_ids)), desc="Reading timeseries data"):
            ts_file = os.path.join(
                ts_dir,
                region,
                basin_ids[k] + ".csv",
            )
            ts_data = pd.read_csv(ts_file, engine="c")
            date = pd.to_datetime(ts_data["date"]).values
            [_, ind1, ind2] = np.intersect1d(date, t_range, return_indices=True)
            for j in range(len(var_lst)):
                x[k, ind2, j] = ts_data[var_lst[j]][ind1].values
        return x

    def _get_unit_json(self, onebasinid, region_name):
        ancfile4unit = os.path.join(
            self.data_source_description["FLOW_DIR"],
            region_name,
            f"{onebasinid}.nc",
        )
        anc4unit = xr.open_dataset(ancfile4unit)
        data_string = anc4unit.Units

        # Convert string to dictionary
        result_dict = {}
        lines = data_string.strip().split("\n")
        for line in lines:
            key, value = line.split(": ")
            unit = value.split("[")[-1].strip("]")

            # Convert to mm/day if the unit is mm
            if unit == "mm":
                unit = "mm/day"

            result_dict[key] = unit
        dataset_variable_names = list(anc4unit.data_vars.keys())
        default_unit = "dimensionless"
        # Update result_dict for matching variables
        for var in dataset_variable_names:
            matched = False
            for key, unit in result_dict.items():
                if key in var:  # Check if the key is a substring of the variable name
                    result_dict[var] = unit
                    matched = True
                    break
            if not matched:  # If no match is found, assign the default unit
                result_dict[var] = default_unit
        # streamflow's unit is same as prcp, we directly set it to mm/day
        result_dict["streamflow"] = "mm/day"
        return result_dict

    def cache_timeseries_xrdataset(self, checkregion, **kwargs):
        if checkregion is not None:
            regions = self.region_data_name if checkregion == "all" else [checkregion]
            self._check_data(regions)

        if isinstance(self.region_data_name, str):
            region_data_name = [self.region_data_name]
        else:
            region_data_name = self.region_data_name
        for region in region_data_name:
            # all files are too large to read in memory, hence we read them region by region
            site_file = os.path.join(
                self.data_source_description["ATTR_DIR"],
                region,
                "attributes_caravan_" + region + ".csv",
            )
            sites_region = pd.read_csv(site_file, sep=",")
            gage_id_lst = sites_region["gauge_id"].values
            batchsize = kwargs.get("batchsize", 100)
            t_range = kwargs.get("t_range", None)
            if t_range is None:
                t_range = ["1980-01-01", "2023-12-31"]
            times = (
                pd.date_range(start=t_range[0], end=t_range[-1], freq="1D")
                .strftime("%Y-%m-%d %H:%M:%S")
                .tolist()
            )
            variables = self.get_relevant_cols()
            units_info = self._get_unit_json(gage_id_lst[0], region)

            def data_generator(basins, batch_size):
                for i in range(0, len(basins), batch_size):
                    yield basins[i : i + batch_size]

            for basin_batch in data_generator(gage_id_lst, batchsize):
                # we make sure the basin ids are sorted
                assert all(x <= y for x, y in zip(basin_batch, basin_batch[1:]))
                data = self.read_timeseries(
                    basin_ids=basin_batch,
                    t_range_list=t_range,
                )
                dataset = xr.Dataset(
                    data_vars={
                        variables[i]: (
                            ["basin", "time"],
                            data[:, :, i],
                            {"units": units_info[variables[i]]},
                        )
                        for i in range(len(variables))
                    },
                    coords={
                        "basin": basin_batch,
                        "time": pd.to_datetime(times),
                    },
                )

                # Save the dataset to a NetCDF file for the current batch and time unit
                prefix_ = "" if region is None else region + "_"
                batch_file_path = os.path.join(
                    CACHE_DIR,
                    f"caravan_{prefix_}timeseries_batch_{basin_batch[0]}_{basin_batch[-1]}.nc",
                )
                dataset.to_netcdf(batch_file_path)

                # Release memory by deleting the dataset
                del dataset
                del data

    def _check_data(self, regions):
        pbar = tqdm(regions, desc="Start Checking Data...")
        for region in pbar:
            pbar.set_description(f"Processing Region-{region}")
            # all files are too large to read in memory, hence we read them region by region
            site_file = os.path.join(
                self.data_source_description["ATTR_DIR"],
                region,
                "attributes_caravan_" + region + ".csv",
            )
            sites_region = pd.read_csv(site_file, sep=",")
            gage_id_lst = sites_region["gauge_id"].values
            # forcing dir is same as flow dir
            ts_dir = self.data_source_description["FORCING_DIR"]
            # Find matching file paths
            for gage_id in tqdm(gage_id_lst, desc="Check data by Gage"):
                file_path = os.path.join(ts_dir, region, gage_id) + ".nc"
                # Check download data! If any error, save csv file to nc file
                try:
                    a_ncfile_data = xr.open_dataset(file_path).assign_coords(
                        gauge_id=gage_id
                    )
                except Exception as e:
                    # it seems there is sth. wrong with hysets_06444000.nc, hence we trans its csv to nc
                    ts_csv_dir = self.data_source_description["TS_CSV_DIR"]
                    csv_file_path = os.path.join(ts_csv_dir, region, gage_id) + ".csv"
                    if not os.path.isfile(csv_file_path):
                        raise FileNotFoundError(
                            f"No csv file found for {gage_id} in {region}"
                        ) from e
                    _data = pd.read_csv(csv_file_path, sep=",", parse_dates=["date"])
                    non_datetime_columns = _data.select_dtypes(
                        exclude=["datetime64[ns]"]
                    ).columns
                    _data[non_datetime_columns] = _data[non_datetime_columns].astype(
                        "float32"
                    )
                    # we assume the last nc file is ok
                    attrs = a_ncfile_data.attrs
                    the_ncfile_data = xr.Dataset.from_dataframe(
                        _data.set_index(["date"])
                    )
                    the_ncfile_data.attrs = attrs
                    # tf = TimezoneFinder()
                    site_meta_file = os.path.join(
                        self.data_source_description["ATTR_DIR"],
                        region,
                        "attributes_other_" + region + ".csv",
                    )
                    df_metadata = pd.read_csv(site_meta_file, sep=",").set_index(
                        "gauge_id"
                    )
                    lat = df_metadata.loc[
                        df_metadata.index == gage_id, "gauge_lat"
                    ].values[0]
                    lon = df_metadata.loc[
                        df_metadata.index == gage_id, "gauge_lon"
                    ].values[0]
                    the_ncfile_data.attrs["Timezone"] = get_tz(lat=lat, lng=lon)
                    the_ncfile = os.path.join(
                        os.path.dirname(file_path), f"{gage_id}.nc"
                    )
                    the_ncfile_data.to_netcdf(the_ncfile)

    def read_attr_xrdataset(self, gage_id_lst=None, var_lst=None, **kwargs):
        # Define the path to the attributes file
        region_name = self.region_data_name
        if isinstance(region_name, list):
            region_name = "_".join(region_name)
        file_path = os.path.join(
            CACHE_DIR,
            f"caravan_{region_name}_attributes.nc",
        )

        # Open the dataset
        ds = xr.open_dataset(file_path)

        # Select the basins
        ds = ds.sel(basin=gage_id_lst)

        # If relevant columns (attributes) are specified, select them
        if var_lst:
            ds = ds[var_lst]

        return ds

    def read_ts_xrdataset(self, gage_id_lst, t_range, var_lst, **kwargs):
        region_name = self.region_data_name
        if isinstance(region_name, list):
            region_name = "_".join(region_name)
        file_paths = sorted(
            glob.glob(
                os.path.join(CACHE_DIR, f"*caravan_{region_name}_*timeseries*.nc")
            )
        )

        # Open the dataset in a lazy manner using dask
        parallel = kwargs.get("parallel", False)
        combined_ds = xr.open_mfdataset(
            file_paths,
            combine="nested",
            concat_dim="basin",
            parallel=parallel,
        )
        # Check for duplicates in the gauge_id dimension
        gauge_ids = pd.Series(combined_ds.gauge_id.values)
        duplicates = gauge_ids[gauge_ids.duplicated()]

        if not duplicates.empty: 
            logging.error("Duplicate gauge IDs found: %s", duplicates.values)  
            raise ValueError("Duplicate gauge IDs found. Please check your data extraction process.")  
        else:  
            logging.info("No duplicate gauge IDs found.")  

        def extract_unit(variable_name, units_string):
            """Construct a pattern based on the variable name to find the unit"""
            # If the variable is 'streamflow', directly use it for unit extraction
            if variable_name == "streamflow":
                main_name = "streamflow"
            # If the variable starts with 'dewpoint_temperature_2m', use the main name 'temperature_2m' for unit extraction
            elif variable_name.startswith("dewpoint_temperature_2m"):
                main_name = "temperature_2m"
            else:
                main_name = "_".join(variable_name.split("_")[:-1])

            pattern = main_name + r":.*?\[(.*?)\]"
            match = re.search(pattern, units_string)
            return match[1].strip() if match else "unknown"

        # If relevant columns are specified, select them
        if var_lst:
            combined_ds = combined_ds[var_lst]
        if t_range:
            combined_ds = combined_ds.sel(time=slice(*t_range))
        if gage_id_lst:
            combined_ds = combined_ds.sel(basin=gage_id_lst)

        # some units are not recognized by pint_xarray, hence we manually set them
        unit_mapping = {"W/m2": "watt / meter ** 2", "m3/m3": "meter^3/meter^3"}

        for var in combined_ds.data_vars:
            if "units" not in combined_ds[var].attrs:
                unit = _extract_unit(var, combined_ds.attrs["Units"])
                # If the extracted unit is in the mapping dictionary, replace it
                unit = unit_mapping.get(unit, unit)
                combined_ds[var].attrs["units"] = unit

        return combined_ds

    @property
    def streamflow_unit(self):
        return "mm/d"

    def read_area(self, gage_id_lst=None):
        return self.read_attr_xrdataset(gage_id_lst, ["area"])

    def read_mean_prcp(self, gage_id_lst=None, unit="mm/d"):
        return self.read_attr_xrdataset(gage_id_lst, ["p_mean"])


def check_coordinates(ds):
    """We uniformly use basin and time as coordinates

    Parameters
    ----------
    ds : xr.Dataset
        the ds must have basin and time as coordinates

    Raises
    ------
    ValueError
        coords not in ds
    """
    required_coords = ["basin", "time"]
    for coord in required_coords:
        if coord not in ds.coords:
            raise ValueError(f"Missing coordinate: {coord}")


def _extract_unit(variable_name, units_string):
    """Construct a pattern based on the variable name to find the unit"""
    # If the variable is 'streamflow', directly use it for unit extraction
    if variable_name == "streamflow":
        main_name = "streamflow"
    # If the variable starts with '*_temperature_2m', use 'temperature_2m' for unit extraction
    elif variable_name.startswith(
        "dewpoint_temperature_2m"
    ) or variable_name.startswith("temperature_2m"):
        main_name = "temperature_2m"
    else:
        main_name = "_".join(variable_name.split("_")[:-1])

    pattern = main_name + r":.*?\[(.*?)\]"
    match = re.search(pattern, units_string)
    the_unit = match[1].strip() if match else "unknown"
    if the_unit == "unknown" and main_name == "streamflow":
        # it is unknown in GRDC-Caravan, so we manually set it
        the_unit = "mm"
    return the_unit
