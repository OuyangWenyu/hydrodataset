import numpy as np
import xarray as xr
from typing import Optional

from hydroutils import hydro_file
from tqdm import tqdm
from hydrodataset import HydroDataset, StandardVariable
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

    def __init__(
        self,
        data_path: str,
        region: Optional[str] = None,
        download: bool = False,
        cache_path: Optional[str] = None,
    ) -> None:
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

    # get the information of features from dataset file"CAMELS-FR_description"
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
            "default_source": "hydroportail",
            "sources": {
                "hydroportail": {"specific_name": "q_cms_obs", "unit": "L/s"},
                "camelsfr": {"specific_name": "q_mm_obs", "unit": "mm/day"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "SIM2-SAFRAN",
            "sources": {
                "SIM2-SAFRAN": {"specific_name": "pcp_mm", "unit": "mm/day"},
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "SIM2-SAFRAN",
            "sources": {
                "SIM2-SAFRAN": {"specific_name": "airtemp_C_mean", "unit": "°C"},
            },
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "SIM2-SAFRAN",
            "sources": {
                "SIM2-SAFRAN": {"specific_name": "airtemp_C_min", "unit": "°C"},
            },
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "SIM2-SAFRAN",
            "sources": {
                "SIM2-SAFRAN": {"specific_name": "airtemp_C_max", "unit": "°C"},
            },
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "oudin",
            "sources": {
                "oudin": {"specific_name": "pet_mm_ou", "unit": "mm/day"},
                "penman": {"specific_name": "pet_mm_pe", "unit": "mm/day"},
                "penman_monteith": {"specific_name": "pet_mm_pm", "unit": "mm/day"},
            },
        },
        StandardVariable.WIND_SPEED: {
            "default_source": "SIM2-SAFRAN",
            "sources": {
                "SIM2-SAFRAN": {"specific_name": "windspeed_mps", "unit": "m/s"},
            },
        },
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "SIM2-SAFRAN",
            "sources": {
                "SIM2-SAFRAN": {"specific_name": "solrad_wm2", "unit": "J/cm^2"},
            },
        },
        StandardVariable.LONGWAVE_SOLAR_RADIATION: {
            "default_source": "SIM2-SAFRAN",
            "sources": {
                "SIM2-SAFRAN": {"specific_name": "lwdownrad_wm2", "unit": "J/cm^2"},
            },
        },
        StandardVariable.SNOW_WATER_EQUIVALENT: {
            "default_source": "isba_model",
            "sources": {
                "isba_model": {"specific_name": "tsd_swe_isba", "unit": "mm/day"},
            },
        },
        StandardVariable.SOIL_MOISTURE: {
            "default_source": "gr",
            "sources": {
                "gr": {"specific_name": "tsd_swi_gr", "unit": "mm/day"},
                "isba": {"specific_name": "tsd_swi_isba", "unit": "mm/day"},
            },
        },      
        StandardVariable.SPECIFIC_HUMIDITY: {
            "default_source": "isba_model",
            "sources": {
                "isba_model": {"specific_name": "spechum_gkg", "unit": "mm/day"},
            },
        },
    }

