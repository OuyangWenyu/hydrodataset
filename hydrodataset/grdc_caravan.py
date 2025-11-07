import os
from typing import Optional

from aqua_fetch import GRDCCaravan as _AquaFetchGRDCCaravan
from hydrodataset import HydroDataset, StandardVariable


# Define the custom GRDCCaravan class at module level to avoid pickle errors
# Named GRDCCaravan to maintain compatibility with file naming conventions
class GRDCCaravan(_AquaFetchGRDCCaravan):
    """Custom GRDCCaravan class with updated URLs and paths for the new dataset version."""

    # Updated URLs for the new dataset version
    url = {
        "GRDC_Caravan_extension_csv.zip": "https://zenodo.org/records/15349031/files/GRDC_Caravan_extension_csv.zip?download=1",
        "GRDC_Caravan_extension_nc.zip": "https://zenodo.org/records/15349031/files/GRDC_Caravan_extension_nc.zip?download=1",
        "grdc-caravan_data_description.pdf": "https://zenodo.org/records/15349031/files/grdc-caravan_data_description.pdf?download=1",
    }

    def __init__(
        self, path=None, overwrite: bool = False, verbosity: int = 1, **kwargs
    ):
        """Custom initialization that uses new URLs and file names."""
        # Import necessary modules
        from aqua_fetch._backend import xarray as xr_backend

        if xr_backend is None:
            self.ftype = "csv"
            if "GRDC_Caravan_extension_nc.zip" in self.url:
                self.url.pop("GRDC_Caravan_extension_nc.zip")
        else:
            self.ftype = "netcdf"
            if "GRDC_Caravan_extension_csv.zip" in self.url:
                self.url.pop("GRDC_Caravan_extension_csv.zip")

        # Call the grandparent class init (from _RainfallRunoff) directly
        from aqua_fetch.rr.utils import _RainfallRunoff

        _RainfallRunoff.__init__(self, path=path, verbosity=verbosity, **kwargs)

        if not os.path.exists(self.path):
            if self.verbosity > 1:
                print(f"Creating directory {self.path}")
            os.makedirs(self.path)

        from aqua_fetch.utils import download, unzip

        for _file, url in self.url.items():
            fpath = os.path.join(self.path, _file)
            if not os.path.exists(fpath) and not overwrite:
                if self.verbosity > 0:
                    print(f"Downloading {_file} from {url}")
                download(url, outdir=self.path, fname=_file)
                unzip(self.path)
            elif self.verbosity > 0:
                print(f"{_file} at {self.path} already exists")

        # Cache stations and attributes
        self._stations = self.other_attributes().index.to_list()
        self._static_attributes = self._static_data().columns.tolist()
        self._dynamic_attributes = self._read_stn_dyn(
            self.stations()[0]
        ).columns.tolist()

    @property
    def shapefiles_path(self):
        """Custom shapefiles_path with updated directory names."""
        if self.ftype == "csv":
            return os.path.join(
                self.path,
                "GRDC_Caravan_extension_csv",
                "GRDC_Caravan_extension_csv",
                "shapefiles",
                "grdc",
            )
        return os.path.join(
            self.path,
            "GRDC_Caravan_extension_nc",
            "GRDC_Caravan_extension_nc",
            "shapefiles",
            "grdc",
        )

    @property
    def attrs_path(self):
        """Custom attrs_path with updated directory names."""
        if self.ftype == "csv":
            return os.path.join(
                self.path,
                "GRDC_Caravan_extension_csv",
                "GRDC_Caravan_extension_csv",
                "attributes",
                "grdc",
            )
        return os.path.join(
            self.path,
            "GRDC_Caravan_extension_nc",
            "GRDC_Caravan_extension_nc",
            "attributes",
            "grdc",
        )

    @property
    def ts_path(self):
        """Custom ts_path with updated directory names."""
        if self.ftype == "csv":
            return os.path.join(
                self.path,
                "GRDC_Caravan_extension_csv",
                "GRDC_Caravan_extension_csv",
                "timeseries",
                "grdc",
            )

        return os.path.join(
            self.path,
            "GRDC_Caravan_extension_nc",
            "GRDC_Caravan_extension_nc",
            "timeseries",
            self.ftype,
            "grdc",
        )


class GrdcCaravan(HydroDataset):
    """GRDC-Caravan dataset class extending HydroDataset.

    This class provides access to the GRDC-Caravan dataset, which contains
    hydrological and meteorological data for watersheds globally.

    This class uses a custom data reading implementation to support a newer
    dataset version than the one supported by the underlying aquafetch library.
    It overrides the download URLs and provides its own parsing and caching logic.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
    """

    def __init__(
        self,
        data_path: str,
        region: Optional[str] = None,
        download: bool = False,
        cache_path: Optional[str] = None,
    ) -> None:
        """Initialize GRDC-Caravan dataset.

        Args:
            data_path: Path to the GRDC-Caravan data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
            cache_path: Path to the cache directory
        """
        super().__init__(data_path, cache_path=cache_path)
        self.region = region
        self.download = download

        # Instantiate the custom class defined at module level
        self.aqua_fetch = GRDCCaravan(data_path)

    @property
    def _attributes_cache_filename(self):
        return "grdc_caravan_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "grdc_caravan_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1950-01-02", "2023-05-18"]

    # get the information of features from grdc-caravan_data_description.pdf
    # Static variable definitions based on inspected data
    _subclass_static_definitions = {
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
    }

    # Dynamic variable mapping based on inspected data
    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "mm",
            "sources": {
                "mm": {"specific_name": "q_mm_obs", "unit": "mm/day"},
                "cms": {"specific_name": "q_cms_obs", "unit": "m^3/s"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "pcp_mm", "unit": "mm/day"},
            },
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "airtemp_c_2m_max", "unit": "°C"},
                "dewpoint": {
                    "specific_name": "dewpoint_temperature_2m_max",
                    "unit": "°C",
                },
            },
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "airtemp_c_2m_min", "unit": "°C"},
                "dewpoint": {
                    "specific_name": "dewpoint_temperature_2m_min",
                    "unit": "°C",
                },
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "airtemp_c_mean_2m", "unit": "°C"},
                "dewpoint": {
                    "specific_name": "dewpoint_temperature_2m_mean",
                    "unit": "°C",
                },
            },
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "era5_land",
            "sources": {
                "era5_land": {
                    "specific_name": "potential_evaporation_sum_era5_land",
                    "unit": "mm/day",
                },
                "fao_pm": {
                    "specific_name": "potential_evaporation_sum_fao_penman_monteith",
                    "unit": "mm/day",
                },
            },
        },
        StandardVariable.SNOW_WATER_EQUIVALENT_MAX: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "snow_depth_water_equivalent_max",
                    "unit": "m",
                },
            },
        },
        StandardVariable.SNOW_WATER_EQUIVALENT_MIN: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "snow_depth_water_equivalent_min",
                    "unit": "m",
                },
            },
        },
        StandardVariable.SNOW_WATER_EQUIVALENT: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "snow_depth_water_equivalent_mean",
                    "unit": "m",
                },
            },
        },
        StandardVariable.SOLAR_RADIATION_MAX: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "surface_net_solar_radiation_max",
                    "unit": "W/m^2",
                },
            },
        },
        StandardVariable.SOLAR_RADIATION_MIN: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "surface_net_solar_radiation_min",
                    "unit": "W/m^2",
                },
            },
        },
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "surface_net_solar_radiation_mean",
                    "unit": "W/m^2",
                },
            },
        },
        StandardVariable.THERMAL_RADIATION_MAX: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "surface_net_thermal_radiation_max",
                    "unit": "W/m^2",
                },
            },
        },
        StandardVariable.THERMAL_RADIATION_MIN: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "surface_net_thermal_radiation_min",
                    "unit": "W/m^2",
                },
            },
        },
        StandardVariable.THERMAL_RADIATION: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "surface_net_thermal_radiation_mean",
                    "unit": "W/m^2",
                },
            },
        },
        StandardVariable.SURFACE_PRESSURE_MAX: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "surface_pressure_max", "unit": "Pa"},
            },
        },
        StandardVariable.SURFACE_PRESSURE_MIN: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "surface_pressure_min", "unit": "Pa"},
            },
        },
        StandardVariable.SURFACE_PRESSURE: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "surface_pressure_mean", "unit": "Pa"},
            },
        },
        StandardVariable.U_WIND_SPEED_MAX: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "u_component_of_wind_10m_max", "unit": "m/s"},
            },
        },
        StandardVariable.U_WIND_SPEED_MIN: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "u_component_of_wind_10m_min", "unit": "m/s"},
            },
        },
        StandardVariable.U_WIND_SPEED: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "u_component_of_wind_10m_mean",
                    "unit": "m/s",
                },
            },
        },
        StandardVariable.V_WIND_SPEED_MAX: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "v_component_of_wind_10m_max", "unit": "m/s"},
            },
        },
        StandardVariable.V_WIND_SPEED_MIN: {
            "default_source": "era5",
            "sources": {
                "era5": {"specific_name": "v_component_of_wind_10m_min", "unit": "m/s"},
            },
        },
        StandardVariable.V_WIND_SPEED: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "v_component_of_wind_10m_mean",
                    "unit": "m/s",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER1_MAX: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "volumetric_soil_water_layer_1_max",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER1_MIN: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "volumetric_soil_water_layer_1_min",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER1: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "volumetric_soil_water_layer_1_mean",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER2_MAX: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "volumetric_soil_water_layer_2_max",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER2_MIN: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "volumetric_soil_water_layer_2_min",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER2: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "volumetric_soil_water_layer_2_mean",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER3_MAX: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "volumetric_soil_water_layer_3_max",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER3_MIN: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "volumetric_soil_water_layer_3_min",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER3: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "volumetric_soil_water_layer_3_mean",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER4_MAX: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "volumetric_soil_water_layer_4_max",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER4_MIN: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "volumetric_soil_water_layer_4_min",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER4: {
            "default_source": "era5",
            "sources": {
                "era5": {
                    "specific_name": "volumetric_soil_water_layer_4_mean",
                    "unit": "m^3/m^3",
                },
            },
        },
    }
