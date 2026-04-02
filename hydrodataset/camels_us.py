import os
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from aqua_fetch import CAMELS_US as _AquaFetchCAMELS_US
from hydrodataset import HydroDataset, StandardVariable

# Import SEP constant for file path separators
try:
    from aqua_fetch._backend import SEP
except ImportError:
    import os
    SEP = os.sep


# Define custom CAMELS_US class at module level to avoid pickle issues
# Named CAMELS_US to maintain compatibility with file naming conventions
class CAMELS_US(_AquaFetchCAMELS_US):
    """Custom CAMELS_US class with Zenodo mirror URLs and -999 missing value handling."""

    # Override URLs to use Zenodo mirror
    # Note: URLs should NOT include the filename as AquaFetch will append it
    url = {
        'camels_attributes_v2.0.pdf': 'https://zenodo.org/records/15529996/files/',
        'camels_attributes_v2.0.xlsx': 'https://zenodo.org/records/15529996/files/',
        'camels_clim.txt': 'https://zenodo.org/records/15529996/files/',
        'camels_geol.txt': 'https://zenodo.org/records/15529996/files/',
        'camels_hydro.txt': 'https://zenodo.org/records/15529996/files/',
        'camels_name.txt': 'https://zenodo.org/records/15529996/files/',
        'camels_soil.txt': 'https://zenodo.org/records/15529996/files/',
        'camels_topo.txt': 'https://zenodo.org/records/15529996/files/',
        'camels_vege.txt': 'https://zenodo.org/records/15529996/files/',
        'readme.txt': 'https://zenodo.org/records/15529996/files/',
        'basin_timeseries_v1p2_metForcing_obsFlow.zip': 'https://zenodo.org/records/15529996/files/',
        'basin_set_full_res.zip': 'https://zenodo.org/records/15529996/files/',
        'basin_timeseries_v1p2_modelOutput_daymet.zip': 'https://zenodo.org/records/15529996/files/',
        'basin_timeseries_v1p2_modelOutput_maurer.zip': 'https://zenodo.org/records/15529996/files/',
        'basin_timeseries_v1p2_modelOutput_nldas.zip': 'https://zenodo.org/records/15529996/files/',
    }

    def _read_stn_dyn(self, stn: str):
        """
        Override parent method to handle -999 missing values in streamflow data.

        According to readme_streamflow.txt:
        "Streamflow data that are missing are given the streamflow value -999.0"

        This method reads dynamic features (forcing + streamflow) and replaces -999
        values with NaN for proper data handling.
        """
        assert isinstance(stn, str)
        df = None
        dir_name = self.folders[self.data_source]

        # Read forcing data
        for cat in os.listdir(os.path.join(self.dataset_dir, dir_name)):
            cat_dirs = os.listdir(os.path.join(self.dataset_dir, f'{dir_name}{SEP}{cat}'))
            stn_file = f'{stn}_lump_cida_forcing_leap.txt'
            if stn_file in cat_dirs:
                df = pd.read_csv(
                    os.path.join(self.dataset_dir, f'{dir_name}{SEP}{cat}{SEP}{stn_file}'),
                    sep=r"\s+|;|:",
                    skiprows=4,
                    engine='python',
                    names=['Year', 'Mnth', 'Day', 'Hr', 'dayl(s)', 'prcp(mm/day)',
                           'srad(W/m2)', 'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)'],
                )
                df.index = pd.to_datetime(
                    df['Year'].map(str) + '-' + df['Mnth'].map(str) + '-' + df['Day'].map(str)
                )

        # Read streamflow data
        flow_dir = os.path.join(self.dataset_dir, 'usgs_streamflow')
        for cat in os.listdir(flow_dir):
            cat_dirs = os.listdir(os.path.join(flow_dir, cat))
            stn_file = f'{stn}_streamflow_qc.txt'
            if stn_file in cat_dirs:
                fpath = os.path.join(flow_dir, f'{cat}{SEP}{stn_file}')
                q_df = pd.read_csv(
                    fpath,
                    sep=r"\s+",
                    names=['station', 'Year', 'Month', 'Day', 'Flow', 'Flag'],
                    engine='python'
                )
                q_df.index = pd.to_datetime(
                    q_df['Year'].map(str) + '-' + q_df['Month'].map(str) + '-' + q_df['Day'].map(str)
                )

                # Replace -999 missing values with NaN
                q_df['Flow'] = q_df['Flow'].replace(-999.0, np.nan)

        # Concatenate forcing and streamflow data
        stn_df = pd.concat([
            df[['dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)']],
            q_df['Flow']
        ], axis=1)

        # Rename columns according to standard names
        stn_df.rename(columns=self.dyn_map, inplace=True)

        # Apply unit conversion factors
        for col, fact in self.dyn_factors.items():
            if col in stn_df.columns:
                stn_df[col] *= fact

        return stn_df


class CamelsUs(HydroDataset):
    """CAMELS_US dataset class.

    This class is a wrapper around the CAMELS_US class from the `aqua_fetch` package.
    It standardizes the dataset into a NetCDF format for easy use with hydrological models.
    It also includes custom logic to read the PET variable from model output files.
    """

    def __init__(
        self, data_path: str, region: Optional[str] = None, download: bool = False
    ) -> None:
        """Initialize CAMELS_US dataset.

        Args:
            data_path: Path to the CAMELS_US data directory. This is where the data will be stored.
            region: Geographic region identifier (optional, defaults to US).
            download: Whether to download data automatically (not used, handled by aqua_fetch).
        """
        super().__init__(data_path)
        self.region = "US" if region is None else region

        # Instantiate the custom class defined at module level
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

    def _dynamic_features(self) -> list:
        """
        Overrides the base method to include 'PET' as a dynamic feature.
        """
        # Get the default features from the parent class (from aquafetch)
        features = super()._dynamic_features()
        # Add the custom PET and ET variables
        features.extend(["PET", "ET"])
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
                gage_id_lst=gage_id_lst, var_lst=["huc_02"], to_numeric=False
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
            tqdm(gage_id_lst, desc="Read model output data (PET and ET) for CAMELS-US")
        ):
            try:
                huc02_ = huc_df.loc[usgs_id, "huc_02"]
                # Convert to 2-digit string with leading zeros if needed
                huc02_ = f"{int(huc02_):02d}"
            except KeyError:
                print(
                    f"Warning: No HUC attribute found for {usgs_id}, skipping PET and ET reading."
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

        # Now, read the PET and ET data for all basins for the default time range
        print("Reading PET and ET data to add to the cache...")
        gage_id_lst = self.read_object_ids().tolist()
        model_output_data = self.read_camels_us_model_output_data(
            gage_id_lst=gage_id_lst, t_range=self.default_t_range, var_lst=["PET", "ET"]
        )

        cache_file = self.cache_dir.joinpath(self._timeseries_cache_filename)

        # Use a with statement to ensure the dataset is closed before writing
        with xr.open_dataset(cache_file) as ds:
            print(f"Variables in base cache: {list(ds.data_vars.keys())}")
            # Create xarray.DataArrays for PET and ET
            pet_da = xr.DataArray(
                model_output_data[:, :, 0],  # PET data
                coords={"basin": gage_id_lst, "time": ds.time},
                dims=["basin", "time"],
                attrs={"units": "mm/day", "source": "SAC-SMA Model Output"},
                name="PET",
            )
            et_da = xr.DataArray(
                model_output_data[:, :, 1],  # ET data
                coords={"basin": gage_id_lst, "time": ds.time},
                dims=["basin", "time"],
                attrs={"units": "mm/day", "source": "SAC-SMA Model Output"},
                name="ET",
            )
            # Merge PET and ET into the main dataset
            # Load the dataset into memory to avoid issues with lazy loading
            merged_ds = ds.load().merge(pet_da).merge(et_da)

        # Now that the original file is closed, we can safely overwrite it
        print("Saving final cache file with merged PET and ET data...")
        print(f"Variables in merged dataset: {list(merged_ds.data_vars.keys())}")
        merged_ds.to_netcdf(cache_file, mode="w")
        print(f"Successfully saved final cache to: {cache_file}")

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
        StandardVariable.EVAPOTRANSPIRATION: {
            "default_source": "sac-sma",
            "sources": {"sac-sma": {"specific_name": "ET", "unit": "mm/day"}},
        },
    }
