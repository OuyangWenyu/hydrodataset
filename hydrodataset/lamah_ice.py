import os
import xarray as xr
from typing import Union, List, Optional

from hydrodataset import HydroDataset, StandardVariable
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime
from aqua_fetch import LamaHIce as _AquaFetchLamaHIce
from aqua_fetch.utils import check_attributes


# Define custom LamaHIce class at module level to avoid pickle issues
# Named LamaHIce to maintain compatibility with file naming conventions
class LamaHIce(_AquaFetchLamaHIce):
    """
    Custom LamaHIce class that overrides URL and methods for new dataset version
    """

    # Override URL at class level
    url = {
        "LamaH-Ice_Caravan_Extension_v15.zip": "https://www.hydroshare.org/resource/705d69c0f77c48538d83cf383f8c63d6/data/contents/LamaH-Ice_Caravan_Extension_v15.zip",
        "lamah_ice.zip": "https://www.hydroshare.org/resource/705d69c0f77c48538d83cf383f8c63d6/data/contents/lamah_ice.zip",
        "lamah_ice_hourly.zip": "https://www.hydroshare.org/resource/705d69c0f77c48538d83cf383f8c63d6/data/contents/lamah_ice_hourly.zip",
    }

    def __init__(
        self,
        path=None,
        overwrite=False,
        *,
        timestep: str = "D",
        data_type: str = "total_upstrm",
        to_netcdf: bool = False,
        **kwargs,
    ):
        """Override __init__ to handle updated URL structure"""
        # don't download hourly data if timestep is daily
        if timestep == "D" and "lamah_ice_hourly.zip" in self.url:
            self.url.pop("lamah_ice_hourly.zip")
        # Updated: changed key from 'Caravan_extension_lamahice.zip' to 'LamaH-Ice_Caravan_Extension_v15.zip'
        if timestep == "H" and "LamaH-Ice_Caravan_Extension_v15.zip" in self.url:
            self.url.pop("LamaH-Ice_Caravan_Extension_v15.zip")

        # Call parent class __init__
        super().__init__(
            path=path,
            timestep=timestep,
            data_type=data_type,
            overwrite=overwrite,
            to_netcdf=to_netcdf,
            **kwargs,
        )

    def fetch_stn_meteo(self, stn: str, nrows: int = None) -> pd.DataFrame:
        """
        Returns climate/meteorological time series data for one station
        Fixed dtype issues for new dataset version
        """
        fpath = os.path.join(self._clim_ts_path(), f"ID_{stn}.csv")

        # Fixed dtypes: changed solar and thermal radiation columns to float32
        dtypes = {
            "YYYY": np.int32,
            "DD": np.int32,
            "MM": np.int32,
            "2m_temp_max": np.float32,
            "2m_temp_mean": np.float32,
            "2m_temp_min": np.float32,
            "2m_dp_temp_max": np.float32,
            "2m_dp_temp_mean": np.float32,
            "2m_dp_temp_min": np.float32,
            "10m_wind_u": np.float32,
            "10m_wind_v": np.float32,
            "fcst_alb": np.float32,
            "lai_high_veg": np.float32,
            "lai_low_veg": np.float32,
            "swe": np.float32,
            "surf_net_solar_rad_max": np.float32,  # Changed from int32 to float32
            "surf_net_solar_rad_mean": np.float32,  # Changed from int32 to float32
            "surf_net_therm_rad_max": np.float32,  # Changed from int32 to float32
            "surf_net_therm_rad_mean": np.float32,  # Changed from int32 to float32
            "surf_press": np.float32,
            "total_et": np.float32,
            "prec": np.float32,
            "volsw_123": np.float32,
            "volsw_4": np.float32,
            "prec_rav": np.float32,
            "prec_carra": np.float32,
        }

        if not os.path.exists(fpath):
            raise FileNotFoundError(f"File not found: {fpath}")

        df = pd.read_csv(fpath, sep=";", dtype=dtypes, nrows=nrows)

        index = df.apply(
            lambda x: datetime.strptime(
                "{0} {1} {2}".format(
                    x["YYYY"].astype(int), x["MM"].astype(int), x["DD"].astype(int)
                ),
                "%Y %m %d",
            ),
            axis=1,
        )

        if self.timestep == "H":
            df.index = index + pd.to_timedelta(df["HOD"], unit="h")
            for col in ["YYYY", "MM", "DD", "DOY", "hh", "mm", "HOD"]:
                df.pop(col)
        else:
            df.index = pd.to_datetime(index)
            for col in [
                "YYYY",
                "MM",
                "DD",
                "DOY",
            ]:
                df.pop(col)

        return df

    def fetch_static_features(
        self,
        stations: Union[str, list] = "all",
        static_features: Union[str, list] = "all",
    ) -> pd.DataFrame:
        """
        Fetches static features of one or more stations with default 'all'
        """
        df = self.static_data()
        df.index = df.index.astype(str)

        static_features = check_attributes(
            static_features, self.static_features, "static_features"
        )
        stations = check_attributes(stations, self.stations(), "stations")

        df = df.loc[stations, static_features]
        return df


class LamahIce(HydroDataset):
    """LamaHICE dataset class extending HydroDataset.

    This class uses a custom data reading implementation to support a newer
    dataset version than the one supported by the underlying aquafetch library.
    It overrides the download URLs and provides updated methods.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
    """

    def __init__(
        self, data_path: str, region: Optional[str] = None, download: bool = False
    ) -> None:
        """Initialize LamaHICE dataset.

        Args:
            data_path: Path to the LamaHICE data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
        """
        super().__init__(data_path)
        self.region = region
        self.download = download

        # Use the custom LamaHIce class defined at module level
        self.aqua_fetch = LamaHIce(data_path)

    @property
    def _attributes_cache_filename(self):
        return "lamahice_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "lamahice_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1950-01-01", "2021-12-31"]

    # Define standardized static variable mappings
    # Based on aqua_fetch LamaHIce static_map
    # information of features get from pdf  https://www.hydroshare.org/resource/705d69c0f77c48538d83cf383f8c63d6/
    _subclass_static_definitions = {
        "p_mean": {"specific_name": "p_mean_basin", "unit": "mm"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
    }

    # Define standardized dynamic variable mappings
    # Based on aqua_fetch LamaHIce dyn_map
    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "lamah_ice",
            "sources": {"lamah_ice": {"specific_name": "q_cms_obs", "unit": "m^3/s"}},
            "sources": {"carra": {"specific_name": "runoff_carra", "unit": "mm"}},
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "lamah_ice",
            "sources": {"lamah_ice": {"specific_name": "pcp_mm", "unit": "mm"}},
            "sources": {"carra": {"specific_name": "prec_carra", "unit": "mm"}},
            "sources": {"rav": {"specific_name": "prec_rav", "unit": "mm"}},
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "lamah_ice",
            "sources": {
                "lamah_ice": {"specific_name": "airtemp_c_2m_min", "unit": "°C"}
            },
            "sources": {"dp": {"specific_name": "2m_dp_temp_min", "unit": "°C"}},
            "sources": {"carra": {"specific_name": "2m_temp_min_carra", "unit": "°C"}},
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "lamah_ice",
            "sources": {
                "lamah_ice": {"specific_name": "airtemp_c_2m_max", "unit": "°C"}
            },
            "sources": {"dp": {"specific_name": "2m_dp_temp_max", "unit": "°C"}},
            "sources": {"carra": {"specific_name": "2m_temp_max_carra", "unit": "°C"}},
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "lamah_ice",
            "sources": {
                "lamah_ice": {"specific_name": "airtemp_c_mean_2m", "unit": "°C"}
            },
            "sources": {"dp": {"specific_name": "2m_dp_temp_mean", "unit": "°C"}},
            "sources": {"rav": {"specific_name": "2m_temp_rav", "unit": "°C"}},
            "sources": {"carra": {"specific_name": "2m_temp_carra", "unit": "°C"}},
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "lamah_ice",
            "sources": {"lamah_ice": {"specific_name": "pet_mm", "unit": "mm/day"}},
        },
        StandardVariable.EVAPOTRANSPIRATION: {
            "default_source": "rav",
            "sources": {"ref": {"specific_name": "ref_et_mm", "unit": "mm/day"}},
            "sources": {"rav": {"specific_name": "total_et_rav", "unit": "mm/day"}},
            "sources": {"carra": {"specific_name": "total_et_carra", "unit": "mm/day"}},
        },
        StandardVariable.U_WIND_SPEED: {
            "default_source": "lamah_ice",
            "sources": {"lamah_ice": {"specific_name": "10m_wind_u", "unit": "m/s"}},
            "sources": {"rav": {"specific_name": "10m_wind_u_rav", "unit": "m/s"}},
        },
        StandardVariable.V_WIND_SPEED: {
            "default_source": "lamah_ice",
            "sources": {"lamah_ice": {"specific_name": "10m_wind_v", "unit": "m/s"}},
            "sources": {"rav": {"specific_name": "10m_wind_v_rav", "unit": "m/s"}},
        },
        StandardVariable.WIND_SPEED: {
            "default_source": "carra",
            "sources": {
                "carra": {"specific_name": "10m_wind_speed_carra", "unit": "m/s"}
            },
        },
        StandardVariable.WIND_DIR: {
            "default_source": "carra",
            "sources": {
                "carra": {"specific_name": "10m_wind_dir_carra", "unit": "degree"}
            },
        },
        StandardVariable.SNOW_WATER_EQUIVALENT: {
            "default_source": "lamah_ice",
            "sources": {"lamah_ice": {"specific_name": "swe", "unit": "mm"}},
            "sources": {"carra": {"specific_name": "swe_carra", "unit": "mm"}},
        },
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "lamah_ice",
            "sources": {
                "lamah_ice": {
                    "specific_name": "surf_net_solar_rad_mean",
                    "unit": "W/m^2",
                }
            },
            "sources": {
                "rav": {"specific_name": "surf_dwn_solar_rad_rav", "unit": "W/m^2"}
            },
            "sources": {
                "carra": {"specific_name": "surf_net_solar_rad_carra", "unit": "W/m^2"}
            },
            "sources": {
                "dwn_carra": {
                    "specific_name": "surf_dwn_solar_rad_carra",
                    "unit": "W/m^2",
                }
            },
        },
        StandardVariable.SOLAR_RADIATION_MAX: {
            "default_source": "lamah_ice",
            "sources": {
                "lamah_ice": {
                    "specific_name": "surf_net_solar_rad_max",
                    "unit": "W/m^2",
                }
            },
        },
        StandardVariable.THERMAL_RADIATION: {
            "default_source": "lamah_ice",
            "sources": {
                "lamah_ice": {
                    "specific_name": "surf_net_therm_rad_mean",
                    "unit": "W/m^2",
                }
            },
            "sources": {
                "outg": {"specific_name": "surf_outg_therm_rad_rav", "unit": "W/m^2"}
            },
            "sources": {
                "dwn": {"specific_name": "surf_dwn_therm_rad_rav", "unit": "W/m^2"}
            },
            "sources": {
                "carra": {"specific_name": "surf_net_therm_rad_carra", "unit": "W/m^2"}
            },
            "sources": {
                "dwn_carra": {
                    "specific_name": "surf_dwn_therm_rad_carra",
                    "unit": "W/m^2",
                }
            },
        },
        StandardVariable.THERMAL_RADIATION_MAX: {
            "default_source": "lamah_ice",
            "sources": {
                "lamah_ice": {
                    "specific_name": "surf_net_therm_rad_max",
                    "unit": "W/m^2",
                }
            },
        },
        StandardVariable.SURFACE_PRESSURE: {
            "default_source": "lamah_ice",
            "sources": {"lamah_ice": {"specific_name": "surf_press", "unit": "Pa"}},
            "sources": {"rav": {"specific_name": "surf_press_rav", "unit": "Pa"}},
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "lamah_ice",
            "sources": {"lamah_ice": {"specific_name": "pet_mm", "unit": "mm"}},
            "sources": {
                "caravan": {
                    "specific_name": "potential_evaporation_sum_fao_penman_monteith_from_caravan",
                    "unit": "mm/day",
                }
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER1: {
            "default_source": "rav",
            "sources": {"rav": {"specific_name": "volsw_123", "unit": "mm"}},
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER4: {
            "default_source": "rav",
            "sources": {"rav": {"specific_name": "volsw_4", "unit": "mm"}},
        },
        StandardVariable.RELATIVE_HUMIDITY: {
            "default_source": "rav",
            "sources": {"rav": {"specific_name": "2m_qv_rav", "unit": "m/s"}},
            "sources": {"carra": {"specific_name": "2m_rel_hum_carra", "unit": "m/s"}},
        },
        StandardVariable.SPECIFIC_HUMIDITY: {
            "default_source": "carra",
            "sources": {"carra": {"specific_name": "2m_spec_hum_carra", "unit": "m/s"}},
        },
        StandardVariable.GROUND_HEAT_FLUX: {
            "default_source": "rav",
            "sources": {"rav": {"specific_name": "grdflx_rav", "unit": "W/m^2"}},
            "sources": {
                "sens": {
                    "specific_name": "surf_dwn_sens_heat_flux_carra",
                    "unit": "W/m^2",
                }
            },
            "sources": {
                "lat": {
                    "specific_name": "surf_dwn_lat_heat_flux_carra",
                    "unit": "W/m^2",
                }
            },
        },
        StandardVariable.SNOW_SUBLIMATION: {
            "default_source": "carra",
            "sources": {
                "carra": {"specific_name": "snow_sublimation_carra", "unit": "mm"}
            },
        },
        StandardVariable.SOIL_MOISTURE: {
            "default_source": "carra",
            "sources": {"carra": {"specific_name": "percolation_carra", "unit": "mm"}},
        },
    }
