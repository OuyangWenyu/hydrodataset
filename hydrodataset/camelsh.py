from hydrodataset import HydroDataset,StandardVariable
from aqua_fetch import CAMELSH


class Camelsh(HydroDataset):
    """CAMELSH (CAMELS-Hourly) dataset class extending RainfallRunoff.

    This class provides access to the CAMELSH dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    def __init__(self, data_path, region=None, download=False):
        """Initialize CAMELSH dataset.

        Args:
            data_path: Path to the CAMELSH data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)   
        """
        super().__init__(data_path)
        self.region = region
        self.download = download
        self.aqua_fetch = CAMELSH(data_path)

    @property
    def _attributes_cache_filename(self):
        return "camelsh_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camelsh_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1980-01-01", "2024-12-31"]

    _subclass_static_definitions = {
        "huc_02": {"specific_name": "huc_02", "unit": "dimensionless"},
        "gauge_lat": {"specific_name": "lat", "unit": "degree"},
        "gauge_lon": {"specific_name": "long", "unit": "degree"},
        "elev_mean": {"specific_name": "elev_mean", "unit": "m"},
        "slope_mean": {"specific_name": "slope_mkm1", "unit": "m/km"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
        "geol_1st_class": {"specific_name": "geol_1st_class", "unit": "dimensionless"},
        "geol_2nd_class": {"specific_name": "geol_2nd_class", "unit": "dimensionless"},
        "geol_porostiy": {"specific_name": "geol_porostiy", "unit": "dimensionless"},
        "geol_permeability": {"specific_name": "geol_permeability", "unit": "m^2"},
        "frac_forest": {"specific_name": "frac_forest", "unit": "dimensionless"},
        "lai_max": {"specific_name": "lai_max", "unit": "dimensionless"},
        "lai_diff": {"specific_name": "lai_diff", "unit": "dimensionless"},
        "dom_land_cover_frac": {
            "specific_name": "dom_land_cover_frac",
            "unit": "dimensionless",
        },
        "dom_land_cover": {"specific_name": "dom_land_cover", "unit": "dimensionless"},
        "root_depth_50": {"specific_name": "root_depth_50", "unit": "m"},
        "root_depth_99": {"specific_name": "root_depth_99", "unit": "m"},
        "soil_depth_statsgo": {"specific_name": "soil_depth_statsgo", "unit": "m"},
        "soil_porosity": {"specific_name": "soil_porosity", "unit": "dimensionless"},
        "soil_conductivity": {"specific_name": "soil_conductivity", "unit": "cm/hr"},
        "max_water_content": {"specific_name": "max_water_content", "unit": "m"},
        "pet_mean": {"specific_name": "pet_mean", "unit": "mm/day"},
    }
    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "usgs",
            "sources": {"usgs": {"specific_name": "q_cms_obs", "unit": "m^3/s"}},
        },
        # TODO: For maurer and nldas, we have not checked the specific names and units.
        StandardVariable.PRECIPITATION: {
            "default_source": "daymet",
            "sources": {
                "daymet": {"specific_name": "pcp_mm", "unit": "mm/day"},
                "maurer": {"specific_name": "prcp_maurer", "unit": "mm/day"},
                "nldas": {"specific_name": "prcp_nldas", "unit": "mm/day"},
            },
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "daymet",
            "sources": {
                "daymet": {"specific_name": "airtemp_c_max", "unit": "°C"},
                "maurer": {"specific_name": "tmax_maurer", "unit": "°C"},
                "nldas": {"specific_name": "tmax_nldas", "unit": "°C"},
            },
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "daymet",
            "sources": {
                "daymet": {"specific_name": "airtemp_c_min", "unit": "°C"},
                "maurer": {"specific_name": "tmin_maurer", "unit": "°C"},
                "nldas": {"specific_name": "tmin_nldas", "unit": "°C"},
            },
        },
        StandardVariable.DAYLIGHT_DURATION: {
            "default_source": "daymet",
            "sources": {
                "daymet": {"specific_name": "dayl", "unit": "s"},
                "maurer": {"specific_name": "dayl_maurer", "unit": "s"},
                "nldas": {"specific_name": "dayl_nldas", "unit": "s"},
            },
        },
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "daymet",
            "sources": {
                "daymet": {"specific_name": "solrad_wm2", "unit": "W/m^2"},
                "maurer": {"specific_name": "srad_maurer", "unit": "W/m^2"},
                "nldas": {"specific_name": "srad_nldas", "unit": "W/m^2"},
            },
        },
        StandardVariable.SNOW_WATER_EQUIVALENT: {
            "default_source": "daymet",
            "sources": {
                "daymet": {"specific_name": "swe_mm", "unit": "mm/day"},
                "maurer": {"specific_name": "swe_maurer", "unit": "mm/day"},
                "nldas": {"specific_name": "swe_nldas", "unit": "mm/day"},
            },
        },
        StandardVariable.VAPOR_PRESSURE: {
            "default_source": "daymet",
            "sources": {
                "daymet": {"specific_name": "vp_hpa", "unit": "hPa"},
                "maurer": {"specific_name": "vp_maurer", "unit": "hPa"},
                "nldas": {"specific_name": "vp_nldas", "unit": "hPa"},
            },
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "sac-sma",
            "sources": {"sac-sma": {"specific_name": "PET", "unit": "mm/day"}},
        },
    }

































'''
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
'''
