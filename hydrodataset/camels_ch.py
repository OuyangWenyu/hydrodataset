import os
import pandas as pd
import numpy as np
from typing import Optional

from aqua_fetch import CAMELS_CH
from hydrodataset import HydroDataset, StandardVariable
from tqdm import tqdm


class CamelsCh(HydroDataset):
    """CAMELS-CH dataset class extending RainfallRunoff.

    This class provides access to the CAMELS-CH dataset, which contains hourly
    hydrological and meteorological data for various watersheds.

    This class overrides the default CSV reading methods from AquaFetch to use
    comma separators instead of semicolon separators, and updates the download
    URL to the latest Zenodo record.
    """

    # Raw CSV column →?zarr variable name (matches NC cache)
    _COL_MAP = {
        "discharge_vol(m3/s)": "q_cms_obs",
        "discharge_spec(mm/d)": "q_mm_obs",
        "waterlevel(m)": "waterlevel",
        "precipitation(mm/d)": "pcp_mm",
        "temperature_min(degC)": "temperature_min",
        "temperature_mean(degC)": "temperature_mean",
        "temperature_max(degC)": "temperature_max",
        "rel_sun_dur(%)": "rel_sun_dur",
        "swe(mm)": "swe_mm",
    }
    _STATIC_FILES = [
        "CAMELS_CH_climate_attributes_obs.csv",
        "CAMELS_CH_geology_attributes.csv",
        "CAMELS_CH_glacier_attributes.csv",
        "CAMELS_CH_humaninfluence_attributes.csv",
        "CAMELS_CH_hydrogeology_attributes.csv",
        "CAMELS_CH_hydrology_attributes_obs.csv",
        "CAMELS_CH_landcover_attributes.csv",
        "CAMELS_CH_soil_attributes.csv",
        "CAMELS_CH_topographic_attributes.csv",
    ]

    def __init__(
        self,
        uri: str,
        region: Optional[str] = None,
        download: bool = False,
        version: str = "v0.9",
    ) -> None:
        super().__init__(uri)
        self.region = region
        self.download = download
        self.version = version

        if str(uri).startswith("s3://"):
            return

        # Define updated URL for the new dataset version
        new_url = "https://zenodo.org/records/15025258"

        # Create custom methods that override the AquaFetch CSV reading
        def custom_climate_attrs(self) -> pd.DataFrame:
            """Returns 14 climate attributes of catchments with comma separator."""
            df = pd.read_csv(
                self.clim_attr_path,
                skiprows=1,
                sep=",",  # Changed from ';' to ','
                index_col="gauge_id",
                dtype={
                    "gauge_id": str,
                    "p_mean": float,
                    "aridity": float,
                    "pet_mean": float,
                    "p_seasonality": float,
                    "frac_snow": float,
                    "high_prec_freq": float,
                    "high_prec_dur": float,
                    "high_prec_timing": str,
                    "low_prec_timing": str,
                },
            )
            return df

        def custom_geol_attrs(self) -> pd.DataFrame:
            """15 geological features with comma separator."""
            df = pd.read_csv(
                self.geol_attr_path,
                skiprows=1,
                sep=",",  # Changed from ';' to ','
                index_col="gauge_id",
                dtype=np.float32,
            )
            df.index = df.index.astype(int).astype(str)
            return df

        def custom_glacier_attrs(self) -> pd.DataFrame:
            """Returns a dataframe with glacier attributes using comma separator."""
            df = pd.read_csv(
                self.glacier_attr_path,
                sep=",",  # Changed from ';' to ','
                skiprows=1,
                index_col="gauge_id",
                dtype=np.float32,
            )
            df.index = df.index.astype(int).astype(str)
            return df

        def custom_human_inf_attrs(self) -> pd.DataFrame:
            """14 anthropogenic factors with comma separator."""
            df = pd.read_csv(
                self.hum_inf_attr_path,
                skiprows=1,
                sep=",",  # Changed from ';' to ','
                index_col="gauge_id",
                dtype={
                    "gauge_id": str,
                    "n_inhabitants": int,
                    "dens_inhabitants": float,
                    "hp_count": int,
                    "hp_qturb": float,
                    "hp_inst_turb": float,
                    "hp_max_power": float,
                    "num_reservoir": int,
                    "reservoir_cap": float,
                    "reservoir_he": float,
                    "reservoir_fs": float,
                    "reservoir_irr": float,
                    "reservoir_nousedata": float,
                },
            )
            return df

        def custom_hydrogeol_attrs(self) -> pd.DataFrame:
            """10 hydrogeological factors with comma separator."""
            df = pd.read_csv(
                self.hydrogeol_attr_path,
                skiprows=1,
                sep=",",  # Changed from ';' to ','
                index_col="gauge_id",
                dtype=float,
            )
            df.index = df.index.astype(int).astype(str)
            return df

        def custom_hydrol_attrs(self) -> pd.DataFrame:
            """14 hydrological parameters + 2 useful infos with comma separator."""
            df = pd.read_csv(
                self.hydrol_attr_path,
                skiprows=1,
                sep=",",  # Changed from ';' to ','
                index_col="gauge_id",
                dtype={
                    "gauge_id": str,
                    "sign_number_of_years": int,
                    "q_mean": float,
                    "runoff_ratio": float,
                    "stream_elas": float,
                    "slope_fdc": float,
                    "baseflow_index_landson": float,
                    "hfd_mean": float,
                    "Q5": float,
                    "Q95": float,
                    "high_q_freq": float,
                    "high_q_dur": float,
                    "low_q_freq": float,
                },
            )
            return df

        def custom_landcolover_attrs(self) -> pd.DataFrame:
            """13 landcover parameters with comma separator."""
            return pd.read_csv(
                self.lc_attr_path,
                skiprows=1,
                sep=",",  # Changed from ';' to ','
                index_col="gauge_id",
                dtype={
                    "gauge_id": str,
                    "crop_perc": float,
                    "grass_perc": float,
                    "scrub_perc": float,
                    "dwood_perc": float,
                    "mixed_wood_perc": float,
                    "ewood_perc": float,
                    "wetlands_perc": float,
                    "inwater_perc": float,
                    "ice_perc": float,
                    "loose_rock_perc": float,
                    "rock_perc": float,
                    "urban_perc": float,
                    "dom_land_cover": str,
                },
            )

        def custom_soil_attrs(self) -> pd.DataFrame:
            """80 soil parameters with comma separator."""
            df = pd.read_csv(
                self.soil_attr_path,
                skiprows=1,
                sep=",",  # Changed from ';' to ','
                index_col="gauge_id",
            )
            df.index = df.index.astype(int).astype(str)
            return df

        def custom_topo_attrs(self) -> pd.DataFrame:
            """Topographic parameters with comma separator."""
            df = pd.read_csv(
                self.topo_attr_path,
                skiprows=1,
                sep=",",  # Changed from ';' to ','
                index_col="gauge_id",
                encoding="unicode_escape",
            )
            df.index = df.index.astype(int).astype(str)
            return df

        def custom_static_data(self) -> pd.DataFrame:
            """Concatenate all static attributes without supp_geol_attrs."""
            df = pd.concat(
                [
                    self.climate_attrs(),
                    self.geol_attrs(),
                    # self.supp_geol_attrs(),  # Removed as requested
                    self.glacier_attrs(),
                    self.human_inf_attrs(),
                    self.hydrogeol_attrs(),
                    self.hydrol_attrs(),
                    self.landcolover_attrs(),
                    self.soil_attrs(),
                    self.topo_attrs(),
                ],
                axis=1,
            )
            df.index = df.index.astype(str)
            df.rename(columns=self.static_map, inplace=True)
            return df

        def custom_read_stn_dyn(self, station: str) -> pd.DataFrame:
            """Reads daily dynamic data for one catchment with comma separator."""
            df = pd.read_csv(
                os.path.join(self.dynamic_path, f"CAMELS_CH_obs_based_{station}.csv"),
                sep=",",  # Changed from ';' to ','
                index_col="date",
                parse_dates=True,
                dtype=np.float32,
            )
            df.rename(columns=self.dyn_map, inplace=True)
            return df

        def custom_stations(self) -> list:
            """Returns station ids for catchments with comma separator."""
            stns = pd.read_csv(
                self.glacier_attr_path, sep=",", skiprows=1  # Changed from ';' to ','
            )["gauge_id"].values.tolist()
            return [str(stn) for stn in stns]

        def custom_dynamic_path(self):
            """Return the correct path for dynamic data (timeseries not time_series)."""
            return os.path.join(self.camels_path, "timeseries", "observation_based")

        def do_nothing(self, *args, **kwargs):
            """Placeholder method to disable certain operations."""
            pass

        # Create class attributes dictionary to override CAMELS_CH methods
        class_attrs = {
            "url": new_url,
            "dynamic_path": property(custom_dynamic_path),
            "climate_attrs": custom_climate_attrs,
            "geol_attrs": custom_geol_attrs,
            "glacier_attrs": custom_glacier_attrs,
            "human_inf_attrs": custom_human_inf_attrs,
            "hydrogeol_attrs": custom_hydrogeol_attrs,
            "hydrol_attrs": custom_hydrol_attrs,
            "landcolover_attrs": custom_landcolover_attrs,
            "soil_attrs": custom_soil_attrs,
            "topo_attrs": custom_topo_attrs,
            "_static_data": custom_static_data,
            "_read_stn_dyn": custom_read_stn_dyn,
            "stations": custom_stations,
            "_maybe_to_netcdf": do_nothing,
        }

        # Create custom CAMELS_CH class with overridden methods
        CustomCamelsCh = type("CAMELS_CH", (CAMELS_CH,), class_attrs)

        try:
            self.aqua_fetch = CustomCamelsCh(uri)
        except Exception as e:
            print(e)
            check_zip_extract = False
            # The zip files that should be downloaded for CAMELS-CH
            zip_files = ["camels_ch.zip", "Caravan_extension_CH.zip"]
            for filename in tqdm(zip_files, desc="Checking zip files"):
                # The extracted directory name (without .zip extension)
                extracted_dir = self.data_source_dir.joinpath(
                    "CAMELS_CH", filename[:-4]
                )
                if not extracted_dir.exists():
                    check_zip_extract = True
                    break
            if check_zip_extract:
                from hydroutils import hydro_file

                hydro_file.zip_extract(self.data_source_dir.joinpath("CAMELS_CH"))
            self.aqua_fetch = CustomCamelsCh(uri)

    @property
    def _attributes_cache_filename(self):
        return "camels_ch_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "camels_ch_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1981-01-01", "2020-12-31"]

    def read_object_ids(self) -> np.ndarray:
        uri = str(self.data_source_dir).rstrip("/")
        rel = "CAMELS_CH/camels_ch/camels_ch/static_attributes/CAMELS_CH_glacier_attributes.csv"
        if self._is_cloud():
            fs = self._make_s3fs()
            with fs.open(f"{uri}/{rel}".removeprefix("s3://")) as fh:
                df = pd.read_csv(fh, comment="#")
        else:
            df = pd.read_csv(os.path.join(uri, *rel.split("/")), comment="#")
        return np.array(sorted(df["gauge_id"].astype(str).tolist()))

    def cache_attributes_to_zarr(self) -> None:
        import zarr
        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")
        attr_base = f"{uri}/CAMELS_CH/camels_ch/camels_ch/static_attributes"
        dfs = []
        for fname in self._STATIC_FILES:
            path = f"{attr_base}/{fname}".removeprefix("s3://")
            try:
                with fs.open(path) as fh:
                    raw = fh.read()
                import io
                for enc in ("utf-8", "latin-1"):
                    try:
                        df = pd.read_csv(io.BytesIO(raw), comment="#",
                                         index_col="gauge_id",
                                         dtype={"gauge_id": str}, encoding=enc)
                        break
                    except UnicodeDecodeError:
                        continue
                df.index = df.index.astype(str)
                dfs.append(df)
            except Exception as e:
                print(f"  WARN {fname}: {e}")
        static = pd.concat(dfs, axis=1)
        static = static.loc[~static.index.duplicated(keep="first")]
        static.columns = self._clean_feature_names(list(static.columns))
        static = static.rename(columns={"area": "area_km2", "gauge_lat": "lat"})

        zarr_name = self._attributes_cache_filename.replace(".nc", ".zarr")
        out, opts = self._zarr_path_and_opts(zarr_name)
        ids = static.index.tolist()
        n = len(ids)
        root = zarr.open_group(out, mode="w", storage_options=opts, zarr_format=2)
        for col in static.columns:
            vals = static[col].values.astype(str) if static[col].dtype == object else static[col].values
            arr = root.create_array(col, shape=(n,), chunks=(n,), dtype=vals.dtype)
            arr[:] = vals
            arr.attrs["_ARRAY_DIMENSIONS"] = ["basin"]
        basin_arr = root.create_array("basin", shape=(n,), chunks=(n,), dtype=str)
        basin_arr[:] = ids
        basin_arr.attrs["_ARRAY_DIMENSIONS"] = ["basin"]
        root.attrs["coordinates"] = "basin"
        self._write_zarr_units(root, "static")
        print(f"Attributes zarr written to: {out}")

    def cache_timeseries_to_zarr(self) -> None:
        import zarr
        fs = self._make_s3fs()
        uri = str(self.data_source_dir).rstrip("/")
        ts_base = f"{uri}/CAMELS_CH/camels_ch/camels_ch/timeseries/observation_based"

        stations = self.read_object_ids().tolist()
        all_times = pd.date_range(self.default_t_range[0], self.default_t_range[1], freq="D")
        n, nt = len(stations), len(all_times)
        times_ns = all_times.asi8
        all_vars = list(self._COL_MAP.values())

        data: dict[str, np.ndarray] = {vn: np.full((n, nt), np.nan) for vn in all_vars}
        print(f"Reading {n} stations from OSS...")
        for i, stn in enumerate(tqdm(stations, desc="CAMELS_CH zarr")):
            path = f"{ts_base}/CAMELS_CH_obs_based_{stn}.csv".removeprefix("s3://")
            try:
                with fs.open(path) as fh:
                    df = pd.read_csv(fh, index_col="date", parse_dates=True)
                df = df.reindex(all_times)
                for raw_col, zarr_vn in self._COL_MAP.items():
                    if raw_col in df.columns:
                        data[zarr_vn][i] = df[raw_col].values.astype(float)
            except Exception as e:
                print(f"  WARN {stn}: {e}")

        zarr_name = self._timeseries_cache_filename.replace(".nc", ".zarr")
        out, opts = self._zarr_path_and_opts(zarr_name)
        root = zarr.open_group(out, mode="w", storage_options=opts, zarr_format=2)
        for vn in all_vars:
            arr = root.create_array(vn, shape=(n, nt), chunks=(min(n, 100), min(nt, 365)), dtype="float64")
            arr[:] = data[vn]
            arr.attrs["_ARRAY_DIMENSIONS"] = ["basin", "time"]

        time_arr = root.create_array("time", shape=(nt,), chunks=(min(nt, 365),), dtype="int64")
        time_arr[:] = times_ns
        time_arr.attrs["_ARRAY_DIMENSIONS"] = ["time"]
        time_arr.attrs["units"] = "nanoseconds since 1970-01-01"
        time_arr.attrs["calendar"] = "proleptic_gregorian"

        basin_arr = root.create_array("basin", shape=(n,), chunks=(n,), dtype=str)
        basin_arr[:] = stations
        basin_arr.attrs["_ARRAY_DIMENSIONS"] = ["basin"]

        root.attrs["coordinates"] = "basin time"
        self._write_zarr_units(root, "dynamic")
        print(f"Timeseries zarr written to: {out}")

    _subclass_static_definitions = {
        "p_mean": {"specific_name": "p_mean", "unit": "mm"},
        "area": {"specific_name": "area_km2", "unit": "km^2"},
    }
    # get the information of features from dataset file"camels_ch_data_description.pdf"
    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "vol",
            "sources": {
                "vol": {"specific_name": "q_cms_obs", "unit": "m^3/s"},
                "spec": {"specific_name": "q_mm_obs", "unit": "mm/d"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "sfo",
            "sources": {
                "sfo": {"specific_name": "pcp_mm", "unit": "mm/day"},
            },
        },
        StandardVariable.TEMPERATURE_MAX: {
            "default_source": "sfo",
            "sources": {"sfo": {"specific_name": "airtemp_C_max", "unit": "°C"}},
        },
        StandardVariable.TEMPERATURE_MIN: {
            "default_source": "sfo",
            "sources": {
                "sfo": {"specific_name": "airtemp_C_min", "unit": "°C"},
            },
        },
        StandardVariable.TEMPERATURE_MEAN: {
            "default_source": "sfo",
            "sources": {
                "sfo": {"specific_name": "airtemp_C_mean", "unit": "°C"},
            },
        },
        StandardVariable.RELATIVE_DAYLIGHT_DURATION: {
            "default_source": "sfo",
            "sources": {
                "sfo": {"specific_name": "rel_sun_dur(%)", "unit": "%"},
            },
        },
        StandardVariable.SNOW_WATER_EQUIVALENT: {
            "default_source": "wsl",
            "sources": {"wsl": {"specific_name": "swe_mm", "unit": "mm"}},
        },
    }
