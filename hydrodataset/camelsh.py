import os
import xarray as xr
from collections import OrderedDict
from hydrodataset import HydroDataset
from tqdm import tqdm
import numpy as np
from pathlib import Path
from water_datasets import CAMELSH


class Camelsh(HydroDataset):
    """CAMELSH (CAMELS-Hourly) dataset class extending RainfallRunoff.

    This class provides access to the CAMELSH dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    def __init__(self, data_path, region=None, download=False, cache_path=None):
        """Initialize CAMELSH dataset.

        Args:
            data_path: Path to the CAMELSH data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
            cache_path: Path to the cache directory
        """
        super().__init__(data_path, cache_path=cache_path)
        self.region = region
        self.download = download
        self.aqua_fetch = CAMELSH(data_path)
        self.set_data_source_describe()

    @property
    def _attributes_cache_filename(self):
        return "camelsh_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camelsh_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1980-01-01", "2024-12-31"]

    def set_data_source_describe(self):
        """Set up dataset file path descriptions.

        Configures paths for various dataset components including timeseries,
        attributes, shapefiles, site info, and hourly data files.
        """
        self.ds_description = OrderedDict()
        self.ds_description["timeseries_dir"] = os.path.join(
            self.data_path,
            "CAMELSH",
            "timeseries",
            "Data",
            "CAMELSH",
            "timeseries",
        )  # timeseries_nonobs
        self.ds_description["attributes_file"] = os.path.join(
            self.data_path, "CAMELSH", "attributes"
        )
        self.ds_description["shapefile_dir"] = os.path.join(
            self.data_path, "CAMELSH", "shapefiles"
        )
        self.ds_description["site_info_file"] = os.path.join(
            self.data_path, "CAMELSH", "info.csv"
        )
        self.ds_description["Hourly2_file"] = os.path.join(
            self.data_path, "CAMELSH", "Hourly2", "Hourly2"
        )





    def _get_attribute_units(self):
        return {
            # 地形特征
            "dis_m3_": "m^3/s",
            "run_mm_": "millimeter",
            "inu_pc_": "percent",
            "lka_pc_": "1e-1 * percent",
            "lkv_mc_": "1e6 * m^3",
            "rev_mc_": "1e6 * m^3",
            "dor_pc_": "percent (x10)",
            "ria_ha_": "hectares",
            "riv_tc_": "1e3 * m^3",
            "gwt_cm_": "centimeter",
            "ele_mt_": "meter",
            "slp_dg_": "1e-1 * degree",
            "sgr_dk_": "decimeter/km",
            "clz_cl_": "dimensionless",
            "cls_cl_": "dimensionless",
            "tmp_dc_": "degree_Celsius",
            "pre_mm_": "millimeters",
            "pet_mm_": "millimeters",
            "aet_mm_": "millimeters",
            "ari_ix_": "1e-2",
            "cmi_ix_": "1e-2",
            "snw_pc_": "percent",
            "glc_cl_": "dimensionless",
            "glc_pc_": "percent",
            "pnv_cl_": "dimensionless",
            "pnv_pc_": "percent",
            "wet_cl_": "dimensionless",
            "wet_pc_": "percent",
            "for_pc_": "percent",
            "crp_pc_": "percent",
            "pst_pc_": "percent",
            "ire_pc_": "percent",
            "gla_pc_": "percent",
            "prm_pc_": "percent",
            "pac_pc_": "percent",
            "tbi_cl_": "dimensionless",
            "tec_cl_": "dimensionless",
            "fmh_cl_": "dimensionless",
            "fec_cl_": "dimensionless",
            "cly_pc_": "percent",
            "slt_pc_": "percent",
            "snd_pc_": "percent",
            "soc_th_": "tonne/hectare",
            "swc_pc_": "percent",
            "lit_cl_": "dimensionless",
            "kar_pc_": "percent",
            "ero_kh_": "kg/hectare/year",
            "pop_ct_": "1e3",
            "ppd_pk_": "1/km^2",
            "urb_pc_": "percent",
            "nli_ix_": "1e-2",
            "rdd_mk_": "meter/km^2",
            "hft_ix_": "1e-1",
            "gad_id_": "dimensionless",
            "gdp_ud_": "dimensionless",
            "hdi_ix_": "1e-3",
        }

    def _get_timeseries_units(self):
        return [
            "mm/day",
            "m",
            "°C",
            "kg/kg",
            "Pa",
            "m/s",
            "m/s",
            "W/m^2",
            "​​Fraction",
            "​​J/kg​​ ",
            "kg/m^2",
            "kg/m^2",
            "W/m²​​ ",
        ]

    @property
    def streamflow_unit(self):
        """Get streamflow unit.

        Returns:
            str: Streamflow unit string
        """
        return "foot^3/s"