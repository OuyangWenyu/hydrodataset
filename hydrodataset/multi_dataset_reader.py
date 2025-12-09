import importlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from hydrodataset import SETTING, StandardVariable

# Dataset mapping
DATASET_MAPPING = {
    "bull": ("hydrodataset.bull", "BULL"),
    "camels_aus": ("hydrodataset.camels_aus", "CamelsAus"),
    "camels_br": ("hydrodataset.camels_br", "CamelsBr"),
    "camels_ch": ("hydrodataset.camels_ch", "CamelsCh"),
    "camels_cl": ("hydrodataset.camels_cl", "CamelsCl"),
    "camels_col": ("hydrodataset.camels_col", "CamelsCol"),
    "camels_de": ("hydrodataset.camels_de", "CamelsDe"),
    "camels_dk": ("hydrodataset.camels_dk", "CamelsDk"),
    "camels_fi": ("hydrodataset.camels_fi", "CamelsFi"),
    "camels_fr": ("hydrodataset.camels_fr", "CamelsFr"),
    "camels_gb": ("hydrodataset.camels_gb", "CamelsGb"),
    "camels_ind": ("hydrodataset.camels_ind", "CamelsInd"),
    "camels_lux": ("hydrodataset.camels_lux", "CamelsLux"),
    "camels_nz": ("hydrodataset.camels_nz", "CamelsNz"),
    "camels_se": ("hydrodataset.camels_se", "CamelsSe"),
    "camels_us": ("hydrodataset.camels_us", "CamelsUs"),
    "camelsh_kr": ("hydrodataset.camelsh_kr", "CamelshKr"),
    "camelsh": ("hydrodataset.camelsh", "Camelsh"),
    "caravan_dk": ("hydrodataset.caravan_dk", "CaravanDK"),
    "grdc_caravan": ("hydrodataset.grdc_caravan", "GrdcCaravan"),
    "hysets": ("hydrodataset.hysets", "Hysets"),
    "lamah_ce": ("hydrodataset.lamah_ce", "LamahCe"),
    "lamah_ice": ("hydrodataset.lamah_ice", "LamahIce"),
    "simbi": ("hydrodataset.simbi", "simbi"),
}


class MultiDatasetReader:
    """Read and manage data from multiple hydrological datasets with ID caching.

    This class provides functionality to:
    - Collect station IDs from multiple datasets
    - Cache IDs for fast subsequent access
    - Detect and handle duplicate IDs across datasets
    - Read timeseries data for specified stations

    Examples
    --------
    >>> from hydrodataset.multi_dataset_reader import MultiDatasetReader
    >>> from hydrodataset import StandardVariable
    >>>
    >>> # Initialize reader
    >>> reader = MultiDatasetReader()
    >>>
    >>> # Collect all IDs (cached automatically)
    >>> id_mapping = reader.collect_all_ids()
    >>>
    >>> # Read data for specific stations
    >>> data = reader.read_data(
    ...     gage_ids=['01013500', '01022500'],
    ...     t_range=['1990-01-01', '2000-12-31'],
    ...     variables=[StandardVariable.STREAMFLOW, StandardVariable.PRECIPITATION]
    ... )
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        datasets: Optional[List[str]] = None,
    ):
        """Initialize the multi-dataset reader.

        Parameters
        ----------
        cache_dir : str, optional
            Directory to store ID cache files. If None, uses SETTING cache directory.
        datasets : list of str, optional
            List of dataset names to use. If None, uses all datasets in DATASET_MAPPING.
        """
        if cache_dir is None:
            self.cache_dir = Path(SETTING["local_data_path"]["cache"])
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Use specified datasets or all available datasets
        if datasets is None:
            self.datasets = list(DATASET_MAPPING.keys())
        else:
            # Validate dataset names
            invalid_datasets = [d for d in datasets if d not in DATASET_MAPPING]
            if invalid_datasets:
                raise ValueError(
                    f"Invalid dataset names: {invalid_datasets}. "
                    f"Available datasets: {list(DATASET_MAPPING.keys())}"
                )
            self.datasets = datasets

        self.id_cache_file = self.cache_dir / "multi_dataset_ids.json"

    def collect_all_ids(self, force_refresh: bool = False) -> Dict[str, List[str]]:
        """Collect all station IDs from specified datasets with deduplication.

        Parameters
        ----------
        force_refresh : bool, optional
            If True, force refresh the cache. Default is False.

        Returns
        -------
        dict
            Dictionary mapping dataset names to lists of station IDs.

        Examples
        --------
        >>> reader = MultiDatasetReader(datasets=['camels_us'])
        >>> id_mapping = reader.collect_all_ids()
        >>> print(f"CAMELS-US has {len(id_mapping['camels_us'])} stations")
        """
        # Load from cache if exists and not forcing refresh
        if self.id_cache_file.exists() and not force_refresh:
            print(f"Loading IDs from cache: {self.id_cache_file}")
            with open(self.id_cache_file, "r") as f:
                return json.load(f)

        print("Collecting station IDs from datasets...")
        id_mapping = {}

        for dataset_name in self.datasets:
            if dataset_name not in DATASET_MAPPING:
                print(f"Warning: Dataset '{dataset_name}' not found in DATASET_MAPPING")
                continue

            try:
                module_name, class_name = DATASET_MAPPING[dataset_name]
                module = importlib.import_module(module_name)
                dataset_class = getattr(module, class_name)

                # Initialize dataset
                data_path = SETTING["local_data_path"]["datasets-origin"]
                ds = dataset_class(data_path)

                # Read station IDs
                ids = ds.read_object_ids()
                id_list = ids.tolist() if isinstance(ids, np.ndarray) else list(ids)

                # Convert all IDs to strings for consistency
                id_list = [str(id) for id in id_list]

                # Remove duplicates within the dataset
                id_list = list(dict.fromkeys(id_list))

                id_mapping[dataset_name] = id_list
                print(f"  {dataset_name}: {len(id_list)} unique IDs")

            except Exception as e:
                print(f"  Error loading {dataset_name}: {e}")
                id_mapping[dataset_name] = []

        # Save to cache
        print(f"\nSaving ID cache to: {self.id_cache_file}")
        with open(self.id_cache_file, "w") as f:
            json.dump(id_mapping, f, indent=2)

        return id_mapping

    def get_global_unique_ids(
        self, id_mapping: Optional[Dict[str, List[str]]] = None
    ) -> tuple[Dict[str, str], Dict[str, List[str]]]:
        """Get globally unique IDs with their source datasets and duplicate information.

        Parameters
        ----------
        id_mapping : dict, optional
            ID mapping from collect_all_ids(). If None, loads from cache.

        Returns
        -------
        tuple of (dict, dict)
            - unique_ids: Dictionary mapping unique IDs to their source dataset names.
            - duplicates: Dictionary mapping duplicate IDs to list of datasets they appear in.

        Examples
        --------
        >>> reader = MultiDatasetReader()
        >>> unique_ids, duplicates = reader.get_global_unique_ids()
        >>> print(f"Total unique IDs: {len(unique_ids)}")
        >>> print(f"Duplicate IDs: {len(duplicates)}")
        """
        if id_mapping is None:
            id_mapping = self.collect_all_ids()

        unique_ids = {}
        duplicates = {}

        for dataset_name, ids in id_mapping.items():
            for id in ids:
                if id in unique_ids:
                    # Track duplicates
                    if id not in duplicates:
                        duplicates[id] = [unique_ids[id]]
                    duplicates[id].append(dataset_name)
                else:
                    unique_ids[id] = dataset_name

        if duplicates:
            print(f"\nWarning: Found {len(duplicates)} duplicate IDs across datasets:")
            for id, datasets in list(duplicates.items())[:5]:  # Show first 5
                print(f"  ID '{id}' appears in: {', '.join(datasets)}")
            if len(duplicates) > 5:
                print(f"  ... and {len(duplicates) - 5} more")

        return unique_ids, duplicates

    def read_data(
        self,
        gage_ids: List[str],
        t_range: List[str],
        variables: Optional[List[StandardVariable]] = None,
        include_duplicates: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """Read data for specified station IDs across datasets.

        Parameters
        ----------
        gage_ids : list of str
            List of station IDs to read.
        t_range : list of str
            Time range as [start_date, end_date], e.g., ["1990-01-01", "2000-12-31"].
        variables : list of StandardVariable, optional
            Variables to read. Default is [STREAMFLOW, PRECIPITATION, TEMPERATURE_MEAN].
        include_duplicates : bool, optional
            If True, read data for duplicate IDs from all datasets they appear in.
            Default is False (only read from first dataset).

        Returns
        -------
        dict
            Dictionary mapping station IDs (or "id@dataset" for duplicates) to DataFrames.

        Examples
        --------
        >>> reader = MultiDatasetReader(datasets=['camels_us'])
        >>> data = reader.read_data(
        ...     gage_ids=['01013500'],
        ...     t_range=['1990-01-01', '1990-12-31'],
        ... )
        >>> print(data['01013500'].head())
        """
        if variables is None:
            variables = [
                StandardVariable.STREAMFLOW,
                StandardVariable.PRECIPITATION,
                StandardVariable.TEMPERATURE_MEAN,
            ]

        # Get ID to dataset mapping and duplicates
        unique_ids, duplicates = self.get_global_unique_ids()

        # Get full ID mapping for finding duplicates
        id_mapping = self.collect_all_ids()

        # Group IDs by dataset
        dataset_ids = {}

        for id in gage_ids:
            if id not in unique_ids:
                print(f"Warning: ID '{id}' not found in any dataset")
                continue

            if include_duplicates and id in duplicates:
                # Read from all datasets that contain this ID
                for dataset_name in duplicates[id]:
                    if dataset_name not in dataset_ids:
                        dataset_ids[dataset_name] = []
                    dataset_ids[dataset_name].append(id)
            else:
                # Only read from first dataset (unique_ids)
                dataset_name = unique_ids[id]
                if dataset_name not in dataset_ids:
                    dataset_ids[dataset_name] = []
                dataset_ids[dataset_name].append(id)

        # Read data from each dataset
        results = {}
        for dataset_name, ids in dataset_ids.items():
            print(f"\nReading data from {dataset_name} for {len(ids)} stations...")

            try:
                module_name, class_name = DATASET_MAPPING[dataset_name]
                module = importlib.import_module(module_name)
                dataset_class = getattr(module, class_name)

                # Initialize dataset
                data_path = SETTING["local_data_path"]["datasets-origin"]
                ds = dataset_class(data_path)

                # Read timeseries data
                ts_data = ds.read_ts_xrdataset(
                    gage_id_lst=ids,
                    t_range=t_range,
                    var_lst=variables,
                )

                # Convert to DataFrames per station
                for id in ids:
                    station_data = ts_data.sel(basin=id).to_dataframe()

                    # If duplicate and include_duplicates, use "id@dataset" as key
                    if include_duplicates and id in duplicates:
                        key = f"{id}@{dataset_name}"
                        results[key] = station_data
                        print(f"  Station {id} (from {dataset_name}): {station_data.shape}")
                    else:
                        results[id] = station_data
                        print(f"  Station {id}: {station_data.shape}")

            except Exception as e:
                print(f"  Error reading data from {dataset_name}: {e}")

        return results
