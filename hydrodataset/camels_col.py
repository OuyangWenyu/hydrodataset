"""
Author: Yimeng Zhang
Date: 2025-10-19 19:40:08
LastEditTime: 2025-10-19 19:40:19
LastEditors: Wenyu Ouyang
Description: CAMELS_COL dataset class
FilePath: \hydrodataset\hydrodataset\camels_col.py
Copyright (c) 2021-2026 Wenyu Ouyang. All rights reserved.
"""

from aqua_fetch import CAMELS_COL
from hydrodataset import HydroDataset
from hydroutils import hydro_file


class CamelsCol(HydroDataset):
    """CAMELS_COL dataset class extending RainfallRunoff.

    This class provides access to the CAMELS_COL dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    def __init__(self, data_path, region=None, download=False):
        """Initialize CAMELS_COL dataset.

        Args:
            data_path: Path to the CAMELS_COL data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
        """
        super().__init__(data_path)
        self.region = region
        self.download = download
        try:
            self.aqua_fetch = CAMELS_COL(data_path)
        except Exception:
            check_zip_extract = False
            zip_files = [
                "01_CAMELS_COL_Attributes.zip",
                "03_CAMELS_COL_Basin_boundary.zip",
                "04_CAMELS_COL_Hydrometeorological_data.zip",
            ]
            for filename in tqdm(zip_files, desc="Checking zip files"):
                extracted_dir = self.data_source_dir.joinpath(
                    "CAMELS_COL", filename[:-4]
                )
                if not extracted_dir.exists():
                    check_zip_extract = True
                    break
            if check_zip_extract:
                hydro_file.zip_extract(self.data_source_dir.joinpath("CAMELS_COL"))
            self.aqua_fetch = CAMELS_COL(data_path)

    @property
    def _attributes_cache_filename(self):
        return "camels_col_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camels_col_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1981-05-21", "2022-12-31"]

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
            "mm/dia",  # pcp_mm
            "mm/dia",  # pet_mm
            "°C",  # airtemp_C_max
            "°C",  # airtemp_C_min
            "°C",  # airtemp_C_mean
            "m^3/s",  # q_cms_obs
        ]
