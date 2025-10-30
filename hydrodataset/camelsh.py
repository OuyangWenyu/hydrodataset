import os
import xarray as xr
import numpy as np
from tqdm import tqdm
from hydrodataset import HydroDataset, StandardVariable
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
        # Basic station information
        "area": {"specific_name": "area_km2", "unit": "km^2"},
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
    }
    _dynamic_variable_mapping = {
        # unit in aquafetch is m^3/s.in paper is kg/m^2
        StandardVariable.STREAMFLOW: {
            "default_source": "nldas",
            "sources": {"nldas": {"specific_name": "q_cms_obs", "unit": "kg/m^2"}},
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "nldas",
            "sources": {
                "nldas": {"specific_name": "pcp_mm", "unit": "mm"},
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "nldas",
            "sources": {
                "nldas": {"specific_name": "airtemp_c_mean", "unit": "°C"},
            },
        },
        StandardVariable.LONGWAVE_SOLAR_RADIATION: {
            "default_source": "nldas",
            "sources": {
                "nldas": {"specific_name": "lwdown", "unit": "W/m^2"},
            },
        },
        # Shortwave radiation flux downwards (surface)
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "nldas",
            "sources": {
                "nldas": {"specific_name": "swdown", "unit": "W/m^2"},
            },
        },
        # unit in aquafetch is mm/day.in paper is kg/m^2
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "nldas",
            "sources": {"nldas": {"specific_name": "pet_mm", "unit": "kg/m^2"}},
        },
        StandardVariable.SURFACE_PRESSURE: {
            "default_source": "nldas",
            "sources": {
                "nldas": {"specific_name": "psurf", "unit": "Pa"},
            },
        },
        # 10-meter above ground Zonal wind speed(east to west)
        StandardVariable.WIND_SPEED: {
            "default_source": "nldas",
            "sources": {
                "nldas": {"specific_name": "wind_e", "unit": "m/s"},
            },
        },
        # 10-meter above ground Meridional wind speed(north to south)
        StandardVariable.MERIDIONAL_WIND_SPEED: {
            "default_source": "nldas",
            "sources": {
                "nldas": {"specific_name": "wind_n", "unit": "m/s"},
            },
        },
        StandardVariable.RELATIVE_HUMIDITY: {
            "default_source": "nldas",
            "sources": {
                "nldas": {"specific_name": "qair", "unit": "kg/kg"},
            },
        },
        StandardVariable.WATER_LEVEL: {
            "default_source": "nldas",
            "sources": {
                "nldas": {"specific_name": "water_level", "unit": "m"},
            },
        },
        StandardVariable.CAPE: {
            "default_source": "nldas",
            "sources": {
                "nldas": {"specific_name": "cape", "unit": "J/kg"},
            },
        },
        StandardVariable.CRAINF_FRAC: {
            "default_source": "nldas",
            "sources": {
                "nldas": {"specific_name": "crainf_frac", "unit": "Fraction"},
            },
        },
    }

    def cache_timeseries_xrdataset(self, batch_size=100):
        """
        Cache timeseries data to NetCDF files in batches, each batch saved as a separate file

        Args:
            batch_size: Number of stations to process per batch, default is 100 stations
        """
        if not hasattr(self, "aqua_fetch"):
            raise NotImplementedError("aqua_fetch attribute is required")

        # Build mapping from variable names to units
        unit_lookup = {}
        if hasattr(self, "_dynamic_variable_mapping"):
            for std_name, mapping_info in self._dynamic_variable_mapping.items():
                for source, source_info in mapping_info["sources"].items():
                    unit_lookup[source_info["specific_name"]] = source_info["unit"]

        # Get all station IDs
        gage_id_lst = self.read_object_ids().tolist()
        total_stations = len(gage_id_lst)

        # Get original variable list and clean
        original_var_lst = self.aqua_fetch.dynamic_features
        cleaned_var_lst = self._clean_feature_names(original_var_lst)
        var_name_mapping = dict(zip(original_var_lst, cleaned_var_lst))

        print(f"Start batch processing {total_stations} stations, {batch_size} stations per batch")
        print(f"Total number of batches: {(total_stations + batch_size - 1)//batch_size}")

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Process stations in batches and save independently
        batch_num = 1
        for batch_idx in range(0, total_stations, batch_size):
            batch_end = min(batch_idx + batch_size, total_stations)
            batch_stations = gage_id_lst[batch_idx:batch_end]

            print(
                f"\nProcessing batch {batch_num}/{(total_stations + batch_size - 1)//batch_size}"
            )
            print(
                f"Station range: {batch_idx} - {batch_end-1} (total {len(batch_stations)} stations)"
            )

            try:
                # Get data for this batch
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

                # Process variables
                new_data_vars = {}
                time_coord = dynamic_data.coords["time"]

                for original_var in tqdm(
                    original_var_lst,
                    desc=f"Processing variables (batch {batch_num})",
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

                # Create Dataset for this batch
                batch_ds = xr.Dataset(
                    data_vars=new_data_vars,
                    coords={
                        "basin": batch_stations,
                        "time": time_coord,
                    },
                )

                # Save this batch to independent file
                batch_filename = f"batch{batch_num:03d}_camelsh_timeseries.nc"
                batch_filepath = self.cache_dir.joinpath(batch_filename)

                print(f"Saving batch {batch_num} to: {batch_filepath}")
                batch_ds.to_netcdf(batch_filepath)
                print(f"Batch {batch_num} saved successfully")

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
    ):
        """
        Read timeseries data (supports standardized variable names and multiple data sources)

        Read data from batch-saved cache files

        Args:
            gage_id_lst: List of station IDs
            t_range: Time range [start, end]
            var_lst: List of standard variable names
            sources: Data source dictionary, format is {variable_name: data_source} or {variable_name: [data_source_list]}

        Returns:
            xr.Dataset: xarray dataset containing requested data
        """
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

        # Process variable name mapping and data source selection
        for std_name in var_lst:
            if std_name not in self._dynamic_variable_mapping:
                raise ValueError(
                    f"'{std_name}' is not a recognized standard variable for this dataset."
                )

            mapping_info = self._dynamic_variable_mapping[std_name]

            # Determine which data source(s) to use
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

            # Only need suffix when user explicitly requests multiple data sources
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

        # Find all batch files
        import glob

        batch_pattern = str(self.cache_dir / "batch*_camelsh_timeseries.nc")
        batch_files = sorted(glob.glob(batch_pattern))

        if not batch_files:
            print("No batch cache files found, starting cache creation...")
            self.cache_timeseries_xrdataset()
            batch_files = sorted(glob.glob(batch_pattern))

            if not batch_files:
                raise FileNotFoundError("Cache creation failed, no batch files found")

        print(f"Found {len(batch_files)} batch files")

        # If no stations specified, read all stations
        if gage_id_lst is None:
            print("No station list specified, will read all stations...")
            gage_id_lst = self.read_object_ids().tolist()

        # Convert station IDs to strings (ensure consistency)
        gage_id_lst = [str(gid) for gid in gage_id_lst]

        # Iterate through batch files to find batches containing required stations
        relevant_datasets = []
        for batch_file in batch_files:
            try:
                # First open only coordinates, don't load data
                ds_batch = xr.open_dataset(batch_file)
                batch_basins = [str(b) for b in ds_batch.basin.values]

                # Check if this batch contains required stations
                common_basins = list(set(gage_id_lst) & set(batch_basins))

                if common_basins:
                    print(
                        f"Batch {os.path.basename(batch_file)}: contains {len(common_basins)} required stations"
                    )

                    # Check if variables exist
                    missing_vars = [
                        v for v in target_vars_to_fetch if v not in ds_batch.data_vars
                    ]
                    if missing_vars:
                        ds_batch.close()
                        raise ValueError(
                            f"Batch {os.path.basename(batch_file)} missing variables: {missing_vars}"
                        )

                    # Select variables and stations
                    ds_subset = ds_batch[target_vars_to_fetch]
                    ds_selected = ds_subset.sel(
                        basin=common_basins, time=slice(t_range[0], t_range[1])
                    )

                    relevant_datasets.append(ds_selected)
                    ds_batch.close()
                else:
                    ds_batch.close()

            except Exception as e:
                print(f"Failed to read batch file {batch_file}: {e}")
                continue

        if not relevant_datasets:
            raise ValueError(f"Specified stations not found in any batch files: {gage_id_lst}")

        print(f"Reading data from {len(relevant_datasets)} batches...")

        # Merge data from all relevant batches
        if len(relevant_datasets) == 1:
            final_ds = relevant_datasets[0]
        else:
            final_ds = xr.concat(relevant_datasets, dim="basin")

        # Rename to standard variable names
        final_ds = final_ds.rename(rename_map)

        # Ensure stations are arranged in input order
        if len(gage_id_lst) > 0:
            # Only select actually existing stations
            existing_basins = [b for b in gage_id_lst if b in final_ds.basin.values]
            if existing_basins:
                final_ds = final_ds.sel(basin=existing_basins)

        return final_ds
