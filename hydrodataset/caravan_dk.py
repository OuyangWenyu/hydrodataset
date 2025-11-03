import os
from aqua_fetch import Caravan_DK
from hydrodataset import HydroDataset, StandardVariable


class CaravanDK(HydroDataset):
    """Caravan_DK dataset class extending HydroDataset.

    This class uses a custom data reading implementation to support a newer
    dataset version than the one supported by the underlying aquafetch library.
    It overrides the download URLs and provides its own parsing and caching logic.
    """

    def __init__(self, data_path, region=None, download=False):
        """Initialize Caravan_DK dataset.

        Args:
            data_path: Path to the Caravan_DK data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
        """
        super().__init__(data_path)
        self.region = region
        self.download = download

        # Define the new URLs for the latest dataset version
        new_url = "https://zenodo.org/records/15200118"

        def do_nothing(self, *args, **kwargs):
            pass

        def custom_boundary_file(self) -> os.PathLike:
            return os.path.join(
                self.path, "shapefiles", "camelsdk", "camelsdk_basin_shapes.shp"
            )

        def custom_csv_path(self):
            return os.path.join(self.path, "timeseries", "csv", "camelsdk")

        def custom_nc_path(self):
            return os.path.join(self.path, "timeseries", "netcdf", "camelsdk")

        def custom_other_attr_fpath(self):
            """returns path to attributes_other_camelsdk.csv file"""
            return os.path.join(
                self.path, "attributes", "camelsdk", "attributes_other_camelsdk.csv"
            )

        def custom_caravan_attr_fpath(self):
            """returns path to attributes_caravan_camelsdk.csv file"""
            return os.path.join(
                self.path, "attributes", "camelsdk", "attributes_caravan_camelsdk.csv"
            )

        def custom_hyd_atlas_fpath(self):
            return os.path.join(
                self.path,
                "attributes",
                "camelsdk",
                "attributes_hydroatlas_camelsdk.csv",
            )

        # Create class attributes dictionary for dynamic class creation
        class_attrs = {
            "url": new_url,
            "boundary_file": property(custom_boundary_file),
            "csv_path": property(custom_csv_path),
            "nc_path": property(custom_nc_path),
            "other_attr_fpath": property(custom_other_attr_fpath),
            "caravan_attr_fpath": property(custom_caravan_attr_fpath),
            "hyd_atlas_fpath": property(custom_hyd_atlas_fpath),
            "_maybe_to_netcdf": do_nothing,
        }

        # Create a custom Caravan_DK class using type() to preserve the class name
        CustomCaravanDK = type("Caravan_DK", (Caravan_DK,), class_attrs)

        # Instantiate our custom class
        self.aqua_fetch = CustomCaravanDK(data_path)

    @property
    def _attributes_cache_filename(self):
        return "caravan_dk_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "caravan_dk_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1981-01-02", "2020-12-31"]

    # Define standardized static variable mappings
    # These variables are already present in the dataset, so we just map them
    # get the information of features from "https://essd.copernicus.org/articles/17/1551/2025/essd-17-1551-2025.pdf"
    _subclass_static_definitions = {
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
    }

    # Define standardized dynamic variable mappings
    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "q_cms_obs", "unit": "m^3/s"}
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "era5_land",
            "sources": {
                "era5_land": {
                    "specific_name": "total_precipitation_sum",
                    "unit": "mm/day",
                }
            },
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "era5_land",
            "sources": {
                "era5_land": {"specific_name": "temperature_2m_max", "unit": "°C"},
                "dewpoint": {
                    "specific_name": "dewpoint_temperature_2m_max",
                    "unit": "°C",
                },
            },
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "era5_land",
            "sources": {
                "era5_land": {"specific_name": "temperature_2m_min", "unit": "°C"},
                "dewpoint": {
                    "specific_name": "dewpoint_temperature_2m_min",
                    "unit": "°C",
                },
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "era5_land",
            "sources": {
                "era5_land": {"specific_name": "temperature_2m_mean", "unit": "°C"},
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
                "fao_penman_monteith": {
                    "specific_name": "potential_evaporation_sum_fao_penman_monteith",
                    "unit": "mm/day",
                },
            },
        },
        StandardVariable.SNOW_WATER_EQUIVALENT: {
            "default_source": "mean",
            "sources": {
                "mean": {
                    "specific_name": "snow_depth_water_equivalent_mean",
                    "unit": "mm",
                },
            },
        },
        StandardVariable.SNOW_WATER_EQUIVALENT_MIN: {
            "default_source": "min",
            "sources": {
                "min": {
                    "specific_name": "snow_depth_water_equivalent_min",
                    "unit": "mm",
                },
            },
        },
        StandardVariable.SNOW_WATER_EQUIVALENT_MAX: {
            "default_source": "max",
            "sources": {
                "max": {
                    "specific_name": "snow_depth_water_equivalent_max",
                    "unit": "mm",
                },
            },
        },
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "mean",
            "sources": {
                "mean": {
                    "specific_name": "surface_net_solar_radiation_mean",
                    "unit": "W/m^2",
                },
            },
        },
        StandardVariable.SOLAR_RADIATION_MIN: {
            "default_source": "min",
            "sources": {
                "min": {
                    "specific_name": "surface_net_solar_radiation_min",
                    "unit": "W/m^2",
                },
            },
        },
        StandardVariable.SOLAR_RADIATION_MAX: {
            "default_source": "max",
            "sources": {
                "max": {
                    "specific_name": "surface_net_solar_radiation_max",
                    "unit": "W/m^2",
                },
            },
        },
        StandardVariable.THERMAL_RADIATION_MIN: {
            "default_source": "min",
            "sources": {
                "min": {
                    "specific_name": "surface_net_thermal_radiation_min",
                    "unit": "W/m^2",
                },
            },
        },
        StandardVariable.THERMAL_RADIATION_MAX: {
            "default_source": "max",
            "sources": {
                "max": {
                    "specific_name": "surface_net_thermal_radiation_max",
                    "unit": "W/m^2",
                },
            },
        },
        StandardVariable.THERMAL_RADIATION: {
            "default_source": "mean",
            "sources": {
                "mean": {
                    "specific_name": "surface_net_thermal_radiation_mean",
                    "unit": "W/m^2",
                },
            },
        },
        StandardVariable.SURFACE_PRESSURE_MIN: {
            "default_source": "min",
            "sources": {
                "min": {"specific_name": "surface_pressure_min", "unit": "Pa"},
            },
        },
        StandardVariable.SURFACE_PRESSURE_MAX: {
            "default_source": "max",
            "sources": {
                "max": {"specific_name": "surface_pressure_max", "unit": "Pa"},
            },
        },
        StandardVariable.SURFACE_PRESSURE: {
            "default_source": "mean",
            "sources": {
                "mean": {"specific_name": "surface_pressure_mean", "unit": "Pa"},
            },
        },
        StandardVariable.U_WIND_SPEED_MIN: {
            "default_source": "min",
            "sources": {
                "min": {"specific_name": "u_component_of_wind_10m_min", "unit": "m/s"},
            },
        },
        StandardVariable.U_WIND_SPEED_MAX: {
            "default_source": "max",
            "sources": {
                "max": {"specific_name": "u_component_of_wind_10m_max", "unit": "m/s"},
            },
        },
        StandardVariable.U_WIND_SPEED: {
            "default_source": "mean",
            "sources": {
                "mean": {
                    "specific_name": "u_component_of_wind_10m_mean",
                    "unit": "m/s",
                },
            },
        },
        StandardVariable.V_WIND_SPEED_MIN: {
            "default_source": "min",
            "sources": {
                "min": {"specific_name": "v_component_of_wind_10m_min", "unit": "m/s"},
            },
        },
        StandardVariable.V_WIND_SPEED_MAX: {
            "default_source": "max",
            "sources": {
                "max": {"specific_name": "v_component_of_wind_10m_max", "unit": "m/s"},
            },
        },
        StandardVariable.V_WIND_SPEED: {
            "default_source": "mean",
            "sources": {
                "mean": {
                    "specific_name": "v_component_of_wind_10m_mean",
                    "unit": "m/s",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER1_MIN: {
            "default_source": "min",
            "sources": {
                "min": {
                    "specific_name": "volumetric_soil_water_layer_1_min",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER1_MAX: {
            "default_source": "max",
            "sources": {
                "max": {
                    "specific_name": "volumetric_soil_water_layer_1_max",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER1: {
            "default_source": "mean",
            "sources": {
                "mean": {
                    "specific_name": "volumetric_soil_water_layer_1_mean",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER2_MIN: {
            "default_source": "min",
            "sources": {
                "min": {
                    "specific_name": "volumetric_soil_water_layer_2_min",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER2_MAX: {
            "default_source": "max",
            "sources": {
                "max": {
                    "specific_name": "volumetric_soil_water_layer_2_max",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER2: {
            "default_source": "mean",
            "sources": {
                "mean": {
                    "specific_name": "volumetric_soil_water_layer_2_mean",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER3_MIN: {
            "default_source": "min",
            "sources": {
                "min": {
                    "specific_name": "volumetric_soil_water_layer_3_min",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER3_MAX: {
            "default_source": "max",
            "sources": {
                "max": {
                    "specific_name": "volumetric_soil_water_layer_3_max",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER3: {
            "default_source": "mean",
            "sources": {
                "mean": {
                    "specific_name": "volumetric_soil_water_layer_3_mean",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER4_MIN: {
            "default_source": "min",
            "sources": {
                "min": {
                    "specific_name": "volumetric_soil_water_layer_4_min",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER4_MAX: {
            "default_source": "max",
            "sources": {
                "max": {
                    "specific_name": "volumetric_soil_water_layer_4_max",
                    "unit": "m^3/m^3",
                },
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER4: {
            "default_source": "mean",
            "sources": {
                "mean": {
                    "specific_name": "volumetric_soil_water_layer_4_mean",
                    "unit": "m^3/m^3",
                },
            },
        },
    }
