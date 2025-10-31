import numpy as np
import xarray as xr

from hydrodataset import HydroDataset, StandardVariable
from tqdm import tqdm
from aqua_fetch import CAMELS_LUX
from hydroutils import hydro_file


class CamelsLux(HydroDataset):
    """CAMELS_LUX dataset class extending RainfallRunoff.

    This class provides access to the CAMELS_LUX dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    def __init__(self, data_path, region=None, download=False):
        """Initialize CAMELS_LUX dataset.

        Args:
            data_path: Path to the CAMELS_LUX data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
        """
        super().__init__(data_path)
        self.region = region
        self.download = download
        try:
            self.aqua_fetch = CAMELS_LUX(data_path)
        except Exception as e:
            print(e)
            check_zip_extract = False
            # The zip files that should be downloaded for CAMELS-LUX
            zip_files = ["CAMELS-LUX.zip", "CAMELS-LUX_shapefiles.zip"]
            for filename in tqdm(zip_files, desc="Checking zip files"):
                extracted_dir = self.data_source_dir.joinpath(
                    "CAMELS_LUX", filename[:-4]
                )
                if not extracted_dir.exists():
                    check_zip_extract = True
                    break
            if check_zip_extract:
                hydro_file.zip_extract(self.data_source_dir.joinpath("CAMELS_LUX"))
            self.aqua_fetch = CAMELS_LUX(data_path)

    @property
    def _attributes_cache_filename(self):
        return "camels_lux_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camels_lux_timeseries.nc"

    @property
    def default_t_range(self):
        return ["2004-01-01", "2021-12-31"]

    def cache_attributes_xrdataset(self):
        """Override base method to add calculated p_mean from precipitation timeseries.

        This method:
        1. Calls parent method to create base attribute cache
        2. Reads precipitation timeseries data
        3. Calculates mean precipitation (p_mean) for each basin
        4. Adds p_mean to the attribute dataset
        5. Saves the updated cache
        """
        # Step 1: Create base attribute cache using parent method
        print("Creating base attribute cache...")
        super().cache_attributes_xrdataset()

        # Step 2: Load the base cache file
        cache_file = self.cache_dir.joinpath(self._attributes_cache_filename)
        with xr.open_dataset(cache_file) as ds_attr:
            ds_attr = ds_attr.load()  # Load into memory

        print("Calculating p_mean from precipitation timeseries...")

        # Step 3: Read precipitation timeseries for all basins
        basin_ids = self.read_object_ids().tolist()

        try:
            # Read full precipitation timeseries
            prcp_ts = self.read_ts_xrdataset(
                gage_id_lst=basin_ids,
                t_range=self.default_t_range,
                var_lst=["precipitation"],
            )

            # Step 4: Calculate temporal mean for each basin
            p_mean_values = prcp_ts["precipitation"].mean(dim="time")

            # Add units attribute
            p_mean_values.attrs["units"] = "mm/day"
            p_mean_values.attrs["description"] = (
                "Mean daily precipitation (calculated from timeseries)"
            )

            # Step 5: Add p_mean to the attribute dataset
            ds_attr["p_mean"] = p_mean_values

            print(f"Successfully calculated p_mean for {len(basin_ids)} basins")

        except Exception as e:
            print(f"Warning: Could not calculate p_mean from precipitation data: {e}")
            print("Creating p_mean with NaN values as placeholder")
            # Create p_mean with NaN values if calculation fails
            p_mean_nan = xr.DataArray(
                np.full(len(basin_ids), np.nan),
                coords={"basin": basin_ids},
                dims=["basin"],
                attrs={
                    "units": "mm/day",
                    "description": "Mean daily precipitation (not available)",
                },
            )
            ds_attr["p_mean"] = p_mean_nan

        # Step 6: Save the updated cache file
        print(f"Saving updated attribute cache with p_mean to: {cache_file}")
        ds_attr.to_netcdf(cache_file, mode="w")
        print("Successfully saved attribute cache with p_mean")

    # get the information of features from dataset file"CAMELS-LUX_data-description.pdf"
    _subclass_static_definitions = {
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
        "gauge_lat": {"specific_name": "lat", "unit": "degree"},
        "gauge_lon": {"specific_name": "long", "unit": "degree"},
        "elev_mean": {"specific_name": "elev_mean", "unit": "m"},
        "pet_mean": {"specific_name": "pet_mean", "unit": "mm/day"},
    }

    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "q_cms_obs", "unit": "m^3/s"},
                "depth_based": {"specific_name": "q_mm_obs", "unit": "mm"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "radar",
            "sources": {
                "radar": {"specific_name": "pcp_mm_radar", "unit": "mm"},
                "station": {"specific_name": "pcp_mm_station", "unit": "mm"},
                "era5": {"specific_name": "pcp_mm_era5", "unit": "mm"},
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": " era5",
            "sources": {
                " era5": {"specific_name": "airtemp_C_mean", "unit": "°C"},
            },
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "oudin",
            "sources": {
                "oudin": {"specific_name": "pet_mm_oudin", "unit": "mm"},
                "penman_monteith": {"specific_name": "pet_mm_pm", "unit": "mm"},
            },
        },
        StandardVariable.RELATIVE_HUMIDITY: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "rh_", "unit": "%"},
            },
        },
        StandardVariable.SPECIFIC_HUMIDITY: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "spechum_gkg", "unit": "kg/kg"},
            },
        },
        StandardVariable.WIND_SPEED: {
            "default_source": "hersbach",
            "sources": {
                "hersbach": {"specific_name": "windspeed_mps", "unit": "m/s"},
            },
        },
        StandardVariable.LOW_LEVEL_WIND_SHEAR: {
            "default_source": "hersbach",
            "sources": {
                "hersbach": {"specific_name": "lls", "unit": "m/s"},
            },
        },
        StandardVariable.DEEP_LEVEL_WIND_SHEAR: {
            "default_source": "hersbach",
            "sources": {
                "hersbach": {"specific_name": "dls", "unit": "m/s"},
            },
        },
        StandardVariable.CAPE: {
            "default_source": "hersbach",
            "sources": {
                "hersbach": {"specific_name": "cape", "unit": "J/kg"},
            },
        },
        StandardVariable.CIN: {
            "default_source": "hersbach",
            "sources": {
                "hersbach": {"specific_name": "cin", "unit": "J/kg"},
            },
        },
        StandardVariable.MAX_RAIN_RATE: {
            "default_source": "radar",
            "sources": {
                "radar": {"specific_name": "rr_max_rad", "unit": "mm/5Min/1x1km"},
            },
        },
        StandardVariable.MIN_RAIN_RATE: {
            "default_source": "radar",
            "sources": {
                "radar": {"specific_name": "rr_min_rad", "unit": "mm/5Min/1x1km"},
            },
        },  
        StandardVariable.CIN: {
            "default_source": "hersbach",
            "sources": {
                "hersbach": {"specific_name": "cin", "unit": "J/kg"},
            },
        },
        StandardVariable.TOTAL_COLUMN_WATER_VAPOUR: {
            "default_source": "hersbach",
            "sources": {
                "hersbach": {"specific_name": "tcwv", "unit": "J/kg"},
            },
        },
        StandardVariable.TOTAL_COLUMN_WATER_VAPOUR: {
            "default_source": "hersbach",
            "sources": {
                "hersbach": {"specific_name": "tcwv", "unit": "J/kg"},
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER1: {
            "default_source": "Muñoz_Sabater",
            "sources": {
                "Muñoz_Sabater": {"specific_name": "sml1", "unit": "m^3/m^3"},
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER2: {
            "default_source": "Muñoz_Sabater",
            "sources": {
                "Muñoz_Sabater": {"specific_name": "sml2", "unit": "m^3/m^3"},
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER3: {
            "default_source": "Muñoz_Sabater",
            "sources": {
                "Muñoz_Sabater": {"specific_name": "sml3", "unit": "m^3/m^3"},
            },
        },
        StandardVariable.VOLUMETRIC_SOIL_WATER_LAYER4: {
            "default_source": "Muñoz_Sabater",
            "sources": {
                "Muñoz_Sabater": {"specific_name": "sml4", "unit": "m^3/m^3"},
            },
        },
    }
