import os
from typing import Optional

import numpy as np
import pandas as pd

from aqua_fetch import CAMELS_PE
from hydrodataset import HydroDataset, StandardVariable


class CamelsPe(HydroDataset):
    """CAMELS-PE dataset reader.

    Thin wrapper over the aqua_fetch ``CAMELS_PE`` class (available since
    aqua-fetch 1.1.0). Data is downloaded/extracted by aqua_fetch to
    ``{root}/CAMELS_PE/CAMELS-PE_v1.0.1/CAMELS-PE/...`` and cached locally as
    ``camels_pe_attributes.nc`` / ``camels_pe_timeseries.nc`` via the base
    ``HydroDataset`` cache methods.

    Cloud (S3/zarr) support is handled by the base class ``cache_*_to_zarr``.
    """

    def __init__(
        self, uri: str, region: Optional[str] = None, download: bool = False
    ) -> None:
        super().__init__(uri)
        self.region = region
        self.download = download
        if not str(uri).startswith("s3://"):
            self.aqua_fetch = CAMELS_PE(uri)

    def get_name(self):
        return "CAMELS_PE"

    def set_data_source_describe(self):
        root = os.path.join(
            str(self.data_source_dir), "CAMELS_PE", "CAMELS-PE_v1.0.1", "CAMELS-PE"
        )
        return {
            "metadata": os.path.join(root, "01_metadata"),
            "attributes": os.path.join(root, "02_attributes"),
            "timeseries": os.path.join(root, "03_timeseries"),
        }

    def download_data_source(self):
        if not hasattr(self, "aqua_fetch"):
            raise NotImplementedError("cloud download is not supported; use cache_*_to_zarr")
        self.aqua_fetch._download_camels_pe(overwrite=True)

    def read_object_ids(self) -> np.ndarray:
        """Read station IDs.

        Local: delegate to the aqua_fetch wrapper's ``stations()``. Cloud (S3):
        read ``stations.csv`` from the metadata dir directly (no aqua_fetch).
        """
        if not self._is_cloud():
            if hasattr(self, "aqua_fetch"):
                return np.sort(np.array(self.aqua_fetch.stations()))
            raise NotImplementedError
        fs = self._make_s3fs()
        rel = "CAMELS_PE/CAMELS-PE_v1.0.1/CAMELS-PE/01_metadata/stations.csv"
        uri = str(self.data_source_dir).rstrip("/")
        with fs.open(f"{uri}/{rel}".removeprefix("s3://")) as fh:
            stations = pd.read_csv(fh, dtype={"gauge_id": str})
        return np.sort(stations["gauge_id"].astype(str).to_numpy())

    def read_target_cols(
        self,
        object_ids=None,
        t_range_list=None,
        target_cols=None,
        gage_id_lst=None,
        t_range=None,
        **kwargs,
    ) -> np.ndarray:
        if object_ids is None:
            object_ids = gage_id_lst
        if t_range_list is None:
            t_range_list = t_range
        if target_cols is None:
            target_cols = ["streamflow"]
        ds = self.read_ts_xrdataset(
            gage_id_lst=object_ids,
            t_range=t_range_list,
            var_lst=target_cols,
            **kwargs,
        )
        return ds.to_array().transpose("basin", "time", "variable").values

    def read_relevant_cols(
        self,
        object_ids=None,
        t_range_list=None,
        relevant_cols=None,
        gage_id_lst=None,
        t_range=None,
        var_lst=None,
        forcing_type=None,
        **kwargs,
    ) -> np.ndarray:
        if object_ids is None:
            object_ids = gage_id_lst
        if t_range_list is None:
            t_range_list = t_range
        if relevant_cols is None:
            relevant_cols = var_lst
        ds = self.read_ts_xrdataset(
            gage_id_lst=object_ids,
            t_range=t_range_list,
            var_lst=relevant_cols,
            **kwargs,
        )
        return ds.to_array().transpose("basin", "time", "variable").values

    def read_constant_cols(
        self,
        object_ids=None,
        constant_cols=None,
        gage_id_lst=None,
        var_lst=None,
        **kwargs,
    ) -> np.ndarray:
        if object_ids is None:
            object_ids = gage_id_lst
        if constant_cols is None:
            constant_cols = var_lst
        ds = self.read_attr_xrdataset(
            gage_id_lst=object_ids,
            var_lst=constant_cols,
            **kwargs,
        )
        return ds.to_array().transpose("basin", "variable").values

    @property
    def _attributes_cache_filename(self):
        return "camels_pe_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camels_pe_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1981-01-01", "2025-12-31"]

    _subclass_static_definitions = {
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
        "gauge_lat": {"specific_name": "lat", "unit": "degree"},
        "gauge_lon": {"specific_name": "long", "unit": "degree"},
        "gauge_elev": {"specific_name": "elev_gauge_m", "unit": "m"},
        "elev_mean": {"specific_name": "elev_catch_m", "unit": "m"},
        "pet_mean": {"specific_name": "pet_mean", "unit": "mm/day"},
        "q_mean": {"specific_name": "q_mean", "unit": "mm/day"},
    }

    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "observations",
            # NOTE: aqua-fetch 1.1.0 exposes only observed streamflow
            # (q_mm_obs); model-simulated flow_sim is intentionally dropped.
            "sources": {
                "observations": {"specific_name": "q_mm_obs", "unit": "mm/day"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "pisco",
            "sources": {
                "pisco": {"specific_name": "pcp_mm", "unit": "mm/day"},
            },
        },
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION: {
            "default_source": "pisco",
            "sources": {
                "pisco": {"specific_name": "pet_mm", "unit": "mm/day"},
            },
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "pisco",
            "sources": {
                "pisco": {"specific_name": "airtemp_c_min", "unit": "degC"},
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "pisco",
            "sources": {
                "pisco": {"specific_name": "airtemp_c_mean", "unit": "degC"},
            },
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "pisco",
            "sources": {
                "pisco": {"specific_name": "airtemp_c_max", "unit": "degC"},
            },
        },
        StandardVariable.SOLAR_RADIATION: {
            "default_source": "era5_land",
            "sources": {
                "era5_land": {"specific_name": "srad", "unit": "MJ/m^2/day"},
            },
        },
        StandardVariable.VAPOR_PRESSURE: {
            "default_source": "era5_land",
            "sources": {
                "era5_land": {"specific_name": "vp_hpa", "unit": "hPa"},
            },
        },
    }
