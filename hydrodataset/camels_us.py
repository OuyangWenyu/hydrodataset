import os
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from aqua_fetch import CAMELS_US
from hydrodataset import HydroDataset


class CamelsUs(HydroDataset):
    """CAMELS_US dataset class.

    This class is a wrapper around the CAMELS_US class from the `aqua_fetch` package.
    It standardizes the dataset into a NetCDF format for easy use with hydrological models.
    It also includes custom logic to read the PET variable from model output files.
    """

    def __init__(self, data_path, region=None, download=False):
        """Initialize CAMELS_US dataset.

        Args:
            data_path: Path to the CAMELS_US data directory. This is where the data will be stored.
            region: Geographic region identifier (optional, defaults to US).
            download: Whether to download data automatically (not used, handled by aqua_fetch).
        """
        super().__init__(data_path)
        self.region = "US" if region is None else region
        self.aqua_fetch = CAMELS_US(data_path)

    @property
    def _attributes_cache_filename(self):
        return "camels_us_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camels_us_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1980-01-01", "2014-12-31"]

    def _get_attribute_units(self):
        """
        Returns a dictionary mapping attribute variables to their units.
        """
        return {
            "gauge_lat": "degree",
            "gauge_lon": "degree",
            "elev_mean": "m",
            "slope_mean": "m/km",
            "area_km2": "km^2",
            "area_geospa_fabric": "km^2",
            "geol_1st_class": "dimensionless",
            "glim_1st_class_frac": "dimensionless",
            "geol_2nd_class": "dimensionless",
            "glim_2nd_class_frac": "dimensionless",
            "carbonate_rocks_frac": "dimensionless",
            "geol_porostiy": "dimensionless",
            "geol_permeability": "m^2",
            "frac_forest": "dimensionless",
            "lai_max": "dimensionless",
            "lai_diff": "dimensionless",
            "gvf_max": "dimensionless",
            "gvf_diff": "dimensionless",
            "dom_land_cover_frac": "dimensionless",
            "dom_land_cover": "dimensionless",
            "root_depth_50": "m",
            "root_depth_99": "m",
            "q_mean": "mm/day",
            "runoff_ratio": "dimensionless",
            "slope_fdc": "dimensionless",
            "baseflow_index": "dimensionless",
            "stream_elas": "dimensionless",
            "q5": "mm/day",
            "q95": "mm/day",
            "high_q_freq": "day/year",
            "high_q_dur": "day",
            "low_q_freq": "day/year",
            "low_q_dur": "day",
            "zero_q_freq": "percent",
            "hfd_mean": "dimensionless",
            "soil_depth_pelletier": "m",
            "soil_depth_statsgo": "m",
            "soil_porosity": "dimensionless",
            "soil_conductivity": "cm/hr",
            "max_water_content": "m",
            "sand_frac": "percent",
            "silt_frac": "percent",
            "clay_frac": "percent",
            "water_frac": "percent",
            "organic_frac": "percent",
            "other_frac": "percent",
            "p_mean": "mm/day",
            "pet_mean": "mm/day",
            "p_seasonality": "dimensionless",
            "frac_snow": "dimensionless",
            "aridity": "dimensionless",
            "high_prec_freq": "days/year",
            "high_prec_dur": "day",
            "high_prec_timing": "dimensionless",
            "low_prec_freq": "days/year",
            "low_prec_dur": "day",
            "low_prec_timing": "dimensionless",
            "huc_02": "dimensionless",
            "gauge_name": "dimensionless",
        }

    def _get_timeseries_units(self):
        """
        Returns a list of units for the time-series variables.
        The order should match the order of variables in `dynamic_features` from aquafetch.
        """
        # Default units from aquafetch, PET will be added separately.
        return ["s", "mm/day", "W/m^2", "mm/day", "°C", "°C", "hPa", "m^3/s", "mm/day"]

    def dynamic_features(self) -> list:
        """
        Overrides the base method to include 'PET' as a dynamic feature.
        """
        # Get the default features from the parent class (from aquafetch)
        features = super().dynamic_features()
        # Add the custom PET variable
        features.append("PET")
        return features

    def read_camels_us_model_output_data(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        forcing_type="daymet",
    ) -> np.array:
        """
        Read model output data of CAMELS-US, including PET.
        This is a legacy function migrated from the old camels.py.
        """
        # Fetch HUC codes for the requested basins on-the-fly
        try:
            huc_ds = self.read_attr_xrdataset(
                gage_id_lst=gage_id_lst, var_lst=["huc_02"]
            )
            huc_df = huc_ds.to_dataframe()
        except Exception as e:
            raise RuntimeError(
                f"Could not read HUC attributes to get model output data: {e}"
            )

        t_range_list = pd.date_range(start=t_range[0], end=t_range[1], freq="D").values
        model_out_put_var_lst = [
            "SWE",
            "PRCP",
            "RAIM",
            "TAIR",
            "PET",
            "ET",
            "MOD_RUN",
            "OBS_RUN",
        ]
        if not set(var_lst).issubset(set(model_out_put_var_lst)):
            raise RuntimeError(
                f"Requested variables not in model output list: {var_lst}"
            )

        nt = len(t_range_list)
        chosen_camels_mods = np.full([len(gage_id_lst), nt, len(var_lst)], np.nan)

        for i, usgs_id in enumerate(
            tqdm(gage_id_lst, desc="Read model output data (PET) for CAMELS-US")
        ):
            try:
                huc02_ = huc_df.loc[usgs_id, "huc_02"]
                # Convert to 2-digit string with leading zeros if needed
                huc02_ = f"{int(huc02_):02d}"
            except KeyError:
                print(
                    f"Warning: No HUC attribute found for {usgs_id}, skipping PET reading."
                )
                continue

            # Construct path to model output files
            file_path_dir = os.path.join(
                self.data_source_dir,
                "CAMELS_US",
                "basin_timeseries_v1p2_modelOutput_" + forcing_type,
                "model_output_" + forcing_type,
                "model_output",
                "flow_timeseries",
                forcing_type,
                huc02_,
            )

            if not os.path.isdir(file_path_dir):
                # This warning is kept for cases where the directory might be missing for a valid HUC
                # print(f"Warning: Model output directory not found: {file_path_dir}")
                continue

            sac_random_seeds = [
                "05",
                "11",
                "27",
                "33",
                "48",
                "59",
                "66",
                "72",
                "80",
                "94",
            ]
            files = [
                os.path.join(file_path_dir, f"{usgs_id}_{seed}_model_output.txt")
                for seed in sac_random_seeds
            ]

            results = []
            for file in files:
                if not os.path.exists(file):
                    continue
                try:
                    result = pd.read_csv(file, sep=r"\s+")
                    df_date = result[["YR", "MNTH", "DY"]]
                    df_date.columns = ["year", "month", "day"]
                    date = pd.to_datetime(df_date).values.astype("datetime64[D]")

                    c, ind1, ind2 = np.intersect1d(
                        date, t_range_list, return_indices=True
                    )
                    if len(c) > 0:
                        temp_data = np.full([nt, len(var_lst)], np.nan)
                        temp_data[ind2, :] = result[var_lst].values[ind1]
                        results.append(temp_data)
                except Exception as e:
                    print(f"Warning: Failed to read {file}: {e}")

            if results:
                result_np = np.array(results)
                # Calculate mean across different random seeds
                with np.errstate(
                    invalid="ignore"
                ):  # Ignore warnings from all-NaN slices
                    chosen_camels_mods[i, :, :] = np.nanmean(result_np, axis=0)

        return chosen_camels_mods

    def cache_timeseries_xrdataset(self):
        """
        Overrides the base method to create a complete cache file including PET.

        This method first calls the parent implementation to create the base cache
        from aquafetch data, then reads the custom PET data and merges it into the
        same cache file.
        """
        # First, create the base cache file using the parent method
        print("Creating base time-series cache from aquafetch...")
        super().cache_timeseries_xrdataset()

        # Now, read the PET data for all basins for the default time range
        print("Reading PET data to add to the cache...")
        gage_id_lst = self.read_object_ids().tolist()
        pet_data = self.read_camels_us_model_output_data(
            gage_id_lst=gage_id_lst, t_range=self.default_t_range, var_lst=["PET"]
        )

        # Squeeze the last dimension if it's 1
        if pet_data.ndim == 3 and pet_data.shape[2] == 1:
            pet_data = np.squeeze(pet_data, axis=2)

        cache_file = self.cache_dir.joinpath(self._timeseries_cache_filename)

        # Use a with statement to ensure the dataset is closed before writing
        with xr.open_dataset(cache_file) as ds:
            # Create an xarray.DataArray for PET
            pet_da = xr.DataArray(
                pet_data,
                coords={"basin": gage_id_lst, "time": ds.time},
                dims=["basin", "time"],
                attrs={"units": "mm/day", "source": "SAC-SMA Model Output"},
                name="PET",
            )
            # Merge PET into the main dataset
            merged_ds = ds.merge(pet_da)

        # Now that the original file is closed, we can safely overwrite it
        print("Saving final cache file with merged PET data...")
        merged_ds.to_netcdf(cache_file, mode="w")
        print(f"Successfully saved final cache to: {cache_file}")

    _variable_mapping = {
        "streamflow": {
            "default_source": "usgs",
            "sources": {"usgs": "q_cms_obs"},
        },
        "precipitation": {
            "default_source": "daymet",
            "sources": {
                "daymet": "pcp_mm",
                "maurer": "prcp_maurer",
                "nldas": "prcp_nldas",
            },
        },
        "temperature_max": {
            "default_source": "daymet",
            "sources": {
                "daymet": "airtemp_C_max",
                "maurer": "tmax_maurer",
                "nldas": "tmax_nldas",
            },
        },
        "temperature_min": {
            "default_source": "daymet",
            "sources": {
                "daymet": "airtemp_C_min",
                "maurer": "tmin_maurer",
                "nldas": "tmin_nldas",
            },
        },
        "daylight_duration": {
            "default_source": "daymet",
            "sources": {
                "daymet": "dayl(s)",
                "maurer": "dayl_maurer",
                "nldas": "dayl_nldas",
            },
        },
        "solar_radiation": {
            "default_source": "daymet",
            "sources": {
                "daymet": "solrad_wm2",
                "maurer": "srad_maurer",
                "nldas": "srad_nldas",
            },
        },
        "snow_water_equivalent": {
            "default_source": "daymet",
            "sources": {
                "daymet": "swe_mm",
                "maurer": "swe_maurer",
                "nldas": "swe_nldas",
            },
        },
        "vapor_pressure": {
            "default_source": "daymet",
            "sources": {
                "daymet": "vp_hpa",
                "maurer": "vp_maurer",
                "nldas": "vp_nldas",
            },
        },
        "potential_evapotranspiration": {
            "default_source": "sac-sma",
            "sources": {"sac-sma": "PET"},
        },
    }
