from typing import Optional

import numpy as np
import xarray as xr
import os
from tqdm import tqdm

from aqua_fetch import HYSETS
from hydrodataset import HydroDataset, StandardVariable


class Hysets(HydroDataset):
    """HYsets dataset class extending RainfallRunoff.

    This class provides access to the HYsets dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    Attributes:
        region: Geographic region identifier
        download: Whether to download data automatically
        ds_description: Dictionary containing dataset file paths
    """

    def __init__(
        self, uri: str, region: Optional[str] = None, download: bool = False
    ) -> None:
        """Initialize HYsets dataset.

        Args:
            uri: Path to the data directory
            region: Geographic region identifier (optional)
            download: Whether to download data automatically (default: False)
        """
        super().__init__(uri)
        self.region = region
        self.download = download
        self.aqua_fetch = HYSETS(uri)

    @property
    def _attributes_cache_filename(self):
        return "hysets_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "hysets_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1950-01-01", "2023-12-31"]

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
            # The result is a DataArray with dimension (basin,)
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

    _subclass_static_definitions = {
        "area": {"specific_name": "area_km2", "unit": "km^2"},
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
    }

    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "observations_cms",
            "sources": {
                "observations_cms": {"specific_name": "q_cms_obs", "unit": "m^3/s"},
                "observations_mm": {"specific_name": "q_mm_obs", "unit": "mm/day"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "observations",
            "sources": {"observations": {"specific_name": "pcp_mm", "unit": "mm/day"}},
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "airtemp_c_2m_max", "unit": "°C"}
            },
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "airtemp_c_2m_min", "unit": "°C"}
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "dptemp_c_mean_2m", "unit": "°C"}
            },
        },
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "solrad_wm2", "unit": "W/m^2"},
                "net": {"specific_name": "solradnet_wm2", "unit": "W/m^2"},
            },
        },
        StandardVariable.EVAPORATION: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "evap_mm", "unit": "mm/day"},
                "snow": {"specific_name": "evap_mm_snow", "unit": "mm/day"},
            },
        },
        StandardVariable.SNOW_WATER_EQUIVALENT: {
            "default_source": "observations",
            "sources": {"observations": {"specific_name": "swe_mm", "unit": "mm"}},
        },
        StandardVariable.SURFACE_PRESSURE: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "airpres_hpa", "unit": "hPa"}
            },
        },
        StandardVariable.U_WIND_SPEED: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "windspeedu_mps", "unit": "m/s"}
            },
        },
        StandardVariable.V_WIND_SPEED: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "windspeedv_mps", "unit": "m/s"}
            },
        },
        StandardVariable.LONGWAVE_SOLAR_RADIATION: {
            "default_source": "downward",
            "sources": {
                "downward": {"specific_name": "lwdownrad_wm2", "unit": "W/m^2"},
                "net": {"specific_name": "lwnetrad_wm2", "unit": "W/m^2"},
            },
        },
        StandardVariable.SNOW_DENSITY: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "snowdensity_kgm3", "unit": "kg/m^3"}
            },
        },
    }

    def cache_timeseries_xrdataset(self, batch_size=1000):
        """Cache timeseries to NetCDF in batches (14425 stations × hourly data)."""
        if not hasattr(self, "aqua_fetch"):
            raise NotImplementedError("aqua_fetch attribute is required")

        unit_lookup = {}
        if hasattr(self, "_dynamic_variable_mapping"):
            for std_name, mapping_info in self._dynamic_variable_mapping.items():
                for source, source_info in mapping_info["sources"].items():
                    unit_lookup[source_info["specific_name"]] = source_info["unit"]

        gage_id_lst = self.read_object_ids().tolist()
        total_stations = len(gage_id_lst)
        original_var_lst = self.aqua_fetch.dynamic_features
        cleaned_var_lst = self._clean_feature_names(original_var_lst)
        var_name_mapping = dict(zip(original_var_lst, cleaned_var_lst))

        n_batches = (total_stations + batch_size - 1) // batch_size
        print(
            f"Start batch processing {total_stations} stations, "
            f"{batch_size} stations per batch ({n_batches} batches)"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        batch_num = 1
        for batch_idx in range(0, total_stations, batch_size):
            batch_end = min(batch_idx + batch_size, total_stations)
            batch_stations = gage_id_lst[batch_idx:batch_end]
            print(
                f"\nProcessing batch {batch_num}/{n_batches} "
                f"(stations {batch_idx}-{batch_end - 1})"
            )
            try:
                batch_data = self.aqua_fetch.fetch_stations_features(
                    stations=batch_stations,
                    dynamic_features=original_var_lst,
                    static_features=None,
                    st=self.default_t_range[0],
                    en=self.default_t_range[1],
                    as_dataframe=False,
                )
                dynamic_data = (
                    batch_data[1] if isinstance(batch_data, tuple) else batch_data
                )

                new_data_vars = {}
                time_coord = dynamic_data.coords["time"]
                for original_var in tqdm(
                    original_var_lst,
                    desc=f"Variables (batch {batch_num})",
                    total=len(original_var_lst),
                ):
                    cleaned_var = var_name_mapping[original_var]
                    var_data = []
                    for station in batch_stations:
                        if station in dynamic_data.data_vars:
                            station_data = dynamic_data[station].sel(
                                dynamic_features=original_var
                            )
                            if "dynamic_features" in station_data.coords:
                                station_data = station_data.drop("dynamic_features")
                            var_data.append(station_data)
                    if var_data:
                        combined = xr.concat(var_data, dim="basin")
                        combined["basin"] = batch_stations
                        combined.attrs["units"] = unit_lookup.get(
                            cleaned_var, "unknown"
                        )
                        new_data_vars[cleaned_var] = combined

                batch_ds = xr.Dataset(
                    data_vars=new_data_vars,
                    coords={"basin": batch_stations, "time": time_coord},
                )
                batch_filepath = self.cache_dir.joinpath(
                    f"batch{batch_num:03d}_hysets_timeseries.nc"
                )
                batch_ds.to_netcdf(batch_filepath)
                print(f"Saved batch {batch_num} -> {batch_filepath}")

            except Exception as e:
                print(f"Batch {batch_num} processing failed: {e}")
                import traceback
                traceback.print_exc()
                continue

            batch_num += 1

        print(f"\nAll batches processed! Total {batch_num - 1} batch files saved")

    def read_ts_xrdataset(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        sources: dict = None,
        **kwargs,
    ) -> xr.Dataset:
        """Read timeseries from the batch-saved cache (standard names + sources)."""
        if (
            not hasattr(self, "_dynamic_variable_mapping")
            or not self._dynamic_variable_mapping
        ):
            raise NotImplementedError(
                "This dataset does not support the standardized variable mapping."
            )
        if var_lst is None:
            var_lst = list(self._dynamic_variable_mapping.keys())
        if t_range is None:
            t_range = self.default_t_range

        target_vars_to_fetch = []
        rename_map = {}
        for std_name in var_lst:
            if std_name not in self._dynamic_variable_mapping:
                raise ValueError(
                    f"'{std_name}' is not a recognized standard variable for this dataset."
                )
            mapping_info = self._dynamic_variable_mapping[std_name]
            is_explicit_source = sources and std_name in sources
            sources_to_use = []
            if is_explicit_source:
                provided_sources = sources[std_name]
                if isinstance(provided_sources, list):
                    sources_to_use.extend(provided_sources)
                else:
                    sources_to_use.append(provided_sources)
            else:
                sources_to_use.append(mapping_info["default_source"])

            needs_suffix = is_explicit_source and len(sources_to_use) > 1
            for source in sources_to_use:
                if source not in mapping_info["sources"]:
                    raise ValueError(
                        f"Source '{source}' is not available for variable '{std_name}'."
                    )
                actual_var_name = mapping_info["sources"][source]["specific_name"]
                target_vars_to_fetch.append(actual_var_name)
                output_name = f"{std_name}_{source}" if needs_suffix else std_name
                rename_map[actual_var_name] = output_name

        import glob

        batch_pattern = str(self.cache_dir / "batch*_hysets_timeseries.nc")
        batch_files = sorted(glob.glob(batch_pattern))
        if not batch_files:
            print("No batch cache files found, starting cache creation...")
            self.cache_timeseries_xrdataset()
            batch_files = sorted(glob.glob(batch_pattern))
            if not batch_files:
                raise FileNotFoundError("Cache creation failed, no batch files found")

        if gage_id_lst is None:
            gage_id_lst = self.read_object_ids().tolist()
        gage_id_lst = [str(gid) for gid in gage_id_lst]

        relevant_datasets = []
        for batch_file in batch_files:
            try:
                ds_batch = xr.open_dataset(batch_file)
                batch_basins = [str(b) for b in ds_batch.basin.values]
                common_basins = list(set(gage_id_lst) & set(batch_basins))
                if common_basins:
                    missing_vars = [
                        v for v in target_vars_to_fetch if v not in ds_batch.data_vars
                    ]
                    if missing_vars:
                        ds_batch.close()
                        raise ValueError(
                            f"Batch {os.path.basename(batch_file)} missing: {missing_vars}"
                        )
                    ds_selected = ds_batch[target_vars_to_fetch].sel(
                        basin=common_basins, time=slice(t_range[0], t_range[1])
                    )
                    relevant_datasets.append(ds_selected)
                ds_batch.close()
            except Exception as e:
                print(f"Failed to read batch file {batch_file}: {e}")
                continue

        if not relevant_datasets:
            raise ValueError(
                f"Specified stations not found in any batch files: {gage_id_lst}"
            )
        if len(relevant_datasets) == 1:
            final_ds = relevant_datasets[0]
        else:
            final_ds = xr.concat(relevant_datasets, dim="basin")

        final_ds = final_ds.rename(rename_map)
        existing_basins = [b for b in gage_id_lst if b in final_ds.basin.values]
        if existing_basins:
            final_ds = final_ds.sel(basin=existing_basins)
        return final_ds
