from hydroutils import hydro_file
from tqdm import tqdm
from hydrodataset import HydroDataset
from aqua_fetch import CAMELS_FR


class CamelsFr(HydroDataset):
    """CAMELS_FR dataset class extending RainfallRunoff.

    This class provides access to the CAMELS_FR dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    def __init__(self, data_path, region=None, download=False, cache_path=None):
        """Initialize CAMELS_FR dataset.

        Args:
            data_path: Path to the CAMELS_FR data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
            cache_path: Path to the cache directory
        """
        super().__init__(data_path, cache_path=cache_path)
        self.region = region
        self.download = download
        try:
            self.aqua_fetch = CAMELS_FR(data_path)
        except Exception as e:
            print(e)
            check_zip_extract = False
            # The zip files that should be downloaded for CAMELS-CH
            zip_files = [
                "ADDITIONAL_LICENSES.zip",
                "CAMELS_FR_attributes.zip",
                "CAMELS_FR_geography.zip",
                "CAMELS_FR_time_series.zip",
            ]
            for filename in tqdm(zip_files, desc="Checking zip files"):
                # The extracted directory name (without .zip extension)
                extracted_dir = self.data_source_dir.joinpath(
                    "CAMELS_FR", filename[:-4]
                )
                if not extracted_dir.exists():
                    check_zip_extract = True
                    break
            if check_zip_extract:
                hydro_file.zip_extract(self.data_source_dir.joinpath("CAMELS_FR"))
            self.aqua_fetch = CAMELS_FR(data_path)

    @property
    def _attributes_cache_filename(self):
        return "camels_fr_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camels_fr_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1970-01-01", "2021-12-31"]

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
            "L/s",  # q_cms_obs
            "mm/day",  # q_mm_obs
            "none",  # tsd_val_s
            "none",  # tsd_val_q
            "none",  # tsd_val_m
            "none",  # tsd_val_c
            "none",  # tsd_val_i
            "mm/day",  # pcp_mm
            "none",  # pcp_mm_solfrac
            "°C",  # airtemp_C_mean
            "mm/day",  # pet_mm_ou
            "mm/day",  # pet_mm_pe
            "mm/day",  # pet_mm_pm
            "m/s",  # windspeed_mps
            "g/kg",  # spechum_gkg
            "J/cm²",  # lwdownrad_wm2
            "J/cm²",  # solrad_wm2
            "none",  # tsd_swi_gr
            "none",  # tsd_swi_isba
            "mm/day",  # tsd_swe_isba
            "°C",  # airtemp_C_min
            "°C",  # airtemp_C_max
        ]
