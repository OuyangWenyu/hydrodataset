from aqua_fetch import CAMELS_DE
from hydrodataset import HydroDataset, StandardVariable
from tqdm import tqdm
from hydroutils import hydro_file


class CamelsDe(HydroDataset):
    """CAMELS-DE dataset class extending RainfallRunoff.

    This class provides access to the CAMELS-DE dataset, which contains hourly
    hydrological and meteorological data for various watersheds.
    """

    def __init__(self, data_path, region=None, download=False, cache_path=None):
        """Initialize CAMELS-DE dataset."""
        super().__init__(data_path, cache_path=cache_path)
        self.region = region
        self.download = download
        try:
            self.aqua_fetch = CAMELS_DE(data_path)
        except Exception:
            check_zip_extract = False
            zip_files = ["camels_de.zip"]
            for filename in tqdm(zip_files, desc="Checking zip files"):
                extracted_dir = self.data_source_dir.joinpath(
                    "CAMELS_DE", filename[:-4]
                )
                if not extracted_dir.exists():
                    check_zip_extract = True
                    break
            if check_zip_extract:
                hydro_file.zip_extract(self.data_source_dir.joinpath("CAMELS_DE"))
            self.aqua_fetch = CAMELS_DE(data_path)

    @property
    def _attributes_cache_filename(self):
        return "camels_de_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camels_de_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1951-01-01", "2020-12-31"]

    # get the information of features from dataset file"CAMELS_DE_Data_Description.pdf"
    _subclass_static_definitions = {
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
        "gauge_lat": {"specific_name": "lat", "unit": "degree"},
        "gauge_lon": {"specific_name": "long", "unit": "degree"},
        "elev_mean": {"specific_name": "elev_mean", "unit": "m"},
    }

    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "vol",
            "sources": {
                "vol": {"specific_name": "q_cms_obs", "unit": "m^3/s"},
                "specific": {"specific_name": "q_mm_obs", "unit": "mm/day"},
            },
        },
        StandardVariable.WATER_LEVEL: {
            "default_source": "federal",
            "sources": {
                "federal": {"specific_name": "water_level", "unit": "m"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "pcp_mm_mean", "unit": "mm/day"},
            },
        },
        StandardVariable.PRECIPITATION_MIN: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "pcp_mm_min", "unit": "mm/day"},
            },
        },
        StandardVariable.PRECIPITATION_MAX: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "pcp_mm_max", "unit": "mm/day"},
            },
        },
        StandardVariable.PRECIPITATION_MEDIAN: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "pcp_mm_median", "unit": "mm/day"},
            },
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "airtemp_c_max", "unit": "°C"},
            },
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "airtemp_c_min", "unit": "°C"},
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "airtemp_c_mean", "unit": "°C"},
            },
        },
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "solrad_wm2_mean", "unit": "W/m^2"},
            },
        },
        StandardVariable.SOLAR_RADIATION_MIN: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "solrad_wm2_min", "unit": "W/m^2"},
            },
        },
        StandardVariable.SOLAR_RADIATION_MAX: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "solrad_wm2_max", "unit": "W/m^2"},
            },
        },
        StandardVariable.SOLAR_RADIATION_MEDIAN: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "solrad_wm2_med", "unit": "W/m^2"},
            },
        },
        StandardVariable.RELATIVE_HUMIDITY: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "rh_", "unit": "%"},
            },
        },
        StandardVariable.RELATIVE_HUMIDITY_MIN: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "rh__min", "unit": "%"},
            },
        },
        StandardVariable.RELATIVE_HUMIDITY_MAX: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "rh__max", "unit": "%"},
            },
        },
        StandardVariable.RELATIVE_HUMIDITY_MEDIAN: {
            "default_source": "dwd",
            "sources": {
                "dwd": {"specific_name": "rh__med", "unit": "%"},
            },
        },
    }
