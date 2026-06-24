"""Read timeseries from multiple datasets via unified station-ID lookup.

Uses the ADR 0001 resolver (``resolve_data_path``) and the built-in
``READER_ALIASES`` / ``_DEFAULT_REGISTRY`` instead of the removed
``SETTING`` global and custom ``DATASET_MAPPING``.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from hydrodataset import StandardVariable
from hydrodataset.configs.data_resolver import READER_ALIASES
from hydrodataset.configs.data_resolver import resolve_data_path
from hydrodataset.configs.settings import get_cache_dir


class MultiDatasetReader:
    """Read data from multiple hydrological datasets with ID caching.

    Collects station IDs across datasets, deduplicates, and reads
    timeseries data by station ID — automatically routing each ID to
    the correct dataset reader.

    Parameters
    ----------
    cache_dir : str, optional
        Directory for the ID-cache JSON file.  Defaults to the value
        returned by ``get_cache_dir()``.
    datasets : list of str, optional
        Dataset ids to use.  Defaults to every dataset that has a
        recognised entry in both ``READER_ALIASES`` and the built-in
        registry (``_DEFAULT_REGISTRY``).
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        datasets: Optional[List[str]] = None,
        source: str = "local",
    ):
        self.source = source
        self.cache_dir = Path(cache_dir if cache_dir is not None else get_cache_dir())
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Build the list of usable dataset ids from the built-in registry
        from hydrodataset.configs.data_resolver import _load_registry  # noqa: F811

        _reg = _load_registry(Path.cwd())

        available = sorted(
            k for k in _reg if k in READER_ALIASES
        )

        if datasets is None:
            self.datasets = available
        else:
            invalid = [d for d in datasets if d not in available]
            if invalid:
                raise ValueError(
                    f"Unknown dataset ids: {invalid}. "
                    f"Available: {available}"
                )
            self.datasets = datasets

        self._id_cache_file = self.cache_dir / "multi_dataset_ids.json"

    # ------------------------------------------------------------------
    # ID collection
    # ------------------------------------------------------------------

    def collect_all_ids(self, force_refresh: bool = False) -> Dict[str, List[str]]:
        """Collect station IDs from all configured datasets.

        Results are written to a JSON cache in *cache_dir* so subsequent
        calls (without *force_refresh*) return instantly.
        """
        if self._id_cache_file.exists() and not force_refresh:
            print(f"Loading IDs from cache: {self._id_cache_file}")
            return json.loads(self._id_cache_file.read_text())

        print("Collecting station IDs from datasets...")
        id_mapping: Dict[str, List[str]] = {}

        for dataset_id in self.datasets:
            try:
                ds = self._get_dataset(dataset_id)
                ids = [str(x) for x in ds.read_object_ids()]
                id_mapping[dataset_id] = list(dict.fromkeys(ids))
                print(f"  {dataset_id}: {len(id_mapping[dataset_id])} unique IDs")
            except Exception as e:
                print(f"  Error loading {dataset_id}: {e}")
                id_mapping[dataset_id] = []

        self._id_cache_file.write_text(json.dumps(id_mapping, indent=2))
        print(f"\nSaved to: {self._id_cache_file}")
        return id_mapping

    def get_global_unique_ids(
        self, id_mapping: Optional[Dict[str, List[str]]] = None
    ):
        """Map every station ID to the first dataset it belongs to.

        Returns
        -------
        unique_ids : dict
            ``{station_id: dataset_id}``
        duplicates : dict
            ``{station_id: [dataset_id, ...]}`` — IDs appearing in
            more than one dataset.
        """
        if id_mapping is None:
            id_mapping = self.collect_all_ids()

        unique_ids: Dict[str, str] = {}
        duplicates: Dict[str, List[str]] = {}

        for dataset_name, ids in id_mapping.items():
            for sid in ids:
                if sid in unique_ids:
                    if sid not in duplicates:
                        duplicates[sid] = [unique_ids[sid]]
                    duplicates[sid].append(dataset_name)
                else:
                    unique_ids[sid] = dataset_name

        if duplicates:
            print(
                f"\nFound {len(duplicates)} duplicate IDs across datasets"
            )
            for sid, dss in list(duplicates.items())[:5]:
                print(f"  ID '{sid}' appears in: {', '.join(dss)}")
            if len(duplicates) > 5:
                print(f"  ... and {len(duplicates) - 5} more")

        return unique_ids, duplicates

    # ------------------------------------------------------------------
    # Data reading
    # ------------------------------------------------------------------

    def read_data(
        self,
        gage_ids: List[str],
        t_range: List[str],
        variables: Optional[List[StandardVariable]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Read timeseries for *gage_ids* across all datasets.

        If a station ID appears in multiple datasets, data from every
        source is returned — keys use the ``id@dataset`` form to avoid
        collisions.
        """
        if variables is None:
            variables = [
                StandardVariable.STREAMFLOW,
                StandardVariable.PRECIPITATION,
                StandardVariable.TEMPERATURE_MEAN,
            ]

        id_mapping = self.collect_all_ids()

        # Build {dataset_id: [id, ...]} by scanning ALL datasets for each id
        dataset_ids: Dict[str, List[str]] = {}
        for sid in gage_ids:
            found = False
            for dname in self.datasets:
                if dname in id_mapping and sid in id_mapping[dname]:
                    dataset_ids.setdefault(dname, []).append(sid)
                    found = True
            if not found:
                print(f"Warning: ID '{sid}' not found in any dataset")

        results: Dict[str, pd.DataFrame] = {}
        for dataset_id, ids in dataset_ids.items():
            print(
                f"\nReading data from {dataset_id} "
                f"for {len(ids)} stations..."
            )
            try:
                ds = self._get_dataset(dataset_id)
                ts_data = ds.read_ts_xrdataset(
                    gage_id_lst=ids,
                    t_range=t_range,
                    var_lst=variables,
                )
                for sid in ids:
                    key = f"{sid}@{dataset_id}"
                    df = ts_data.sel(basin=sid).to_dataframe()
                    results[key] = df
                    print(f"  {key}: {df.shape}")
            except Exception as e:
                print(f"  Error reading from {dataset_id}: {e}")

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_dataset(self, dataset_id: str):
        """Construct a dataset instance for *dataset_id* via the resolver."""
        spec = READER_ALIASES[dataset_id]
        module = importlib.import_module(spec["module"])
        cls = getattr(module, spec["class"])
        return cls(uri=resolve_data_path(dataset_id, source=self.source))
