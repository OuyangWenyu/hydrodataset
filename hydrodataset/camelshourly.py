import collections
import fnmatch
import logging
import os
import tarfile
import numpy as np
import pandas as pd
import xarray as xr

from tqdm import tqdm
from pathlib import Path
from codetiming import Timer

from hydrodataset.camels import Camels


LOGGER = logging.getLogger(__name__)


class CamelsHourly(Camels):
    """A class reading dataset (made by Gauch, Martin et al.) providing hourly data for CAMELS US basins

    This class extends the `Camels` class by hourly in- and output data. Currently, only NLDAS forcings are
    available at an hourly resolution.
    """

    def __init__(
        self,
        data_path,
        download=False,
        region: str = "US",
    ):
        super().__init__(data_path, download, region)

    def get_name(self):
        return "HOURLY_CAMELS_" + self.region

    def download_data_source(self) -> None:
        """
        unzip manually downloaded data

        Returns
        -------
        None
        """
        camels_config = self.data_source_description
        LOGGER.warning(
            f"## Please manually download data from {camels_config['CAMELS_US_HOURLY_DOWNLOAD_URL']} now."
        )
        for f_name in os.listdir(camels_config["CAMELS_US_HOURLY_DIR"]):
            if fnmatch.fnmatch(f_name, "*.tar.gz"):
                # Path to the extracted folder
                extracted_folder_path = os.path.join(
                    camels_config["CAMELS_US_HOURLY_DIR"], f_name[:-7]
                )

                # Check if the folder already exists
                if not os.path.exists(extracted_folder_path):
                    with tarfile.open(
                        os.path.join(camels_config["CAMELS_US_HOURLY_DIR"], f_name)
                    ) as file:
                        # Get the total number of files within the archive
                        total_files = len(file.getmembers())

                        # Extract with progress bar
                        for member in tqdm(
                            file.getmembers(),
                            total=total_files,
                            desc=f"Extracting {f_name}",
                            unit="file",
                        ):
                            file.extract(member, extracted_folder_path)
                else:
                    LOGGER.info(
                        f"Skipping extraction, directory already exists: {extracted_folder_path}"
                    )

    def set_data_source_describe(self) -> collections.OrderedDict:
        """
        the files in the dataset and their location in file system

        Returns
        -------
        collections.OrderedDict
            the description for GPM and GFS dataset
        """
        if self.region != "US":
            raise NotImplementedError(
                "Now only CAMELS US is supported. Please check if you choose correctly:\n"
            )
        camels_us_hourly_root_dir = self.data_source_dir
        camels_us_dir = camels_us_hourly_root_dir.parent.joinpath("camels_us")
        camels_us_gague_file = camels_us_dir.joinpath("camels_name.txt")
        camels_us_attr_dir = camels_us_dir.joinpath(
            "camels_attributes_v2.0", "camels_attributes_v2.0"
        )
        if not camels_us_gague_file.exists():
            raise FileNotFoundError(
                "We use gauge_id_file from camels dataset -- camels_name.txt, "
                f"but it is not found in {camels_us_dir}. "
                "Please check if the file exists."
            )
        if not camels_us_attr_dir.exists():
            raise FileNotFoundError(
                "We use attributes from camels dataset -- camels_attributes_v2.0, "
                f"but it is not found in {camels_us_dir}. "
                "Please check if the file exists."
            )
        return collections.OrderedDict(
            CAMELS_US_HOURLY_DIR=camels_us_hourly_root_dir,
            CAMELS_US_HOURLY_TS_NC=camels_us_hourly_root_dir.joinpath(
                "usgs-streamflow-nldas_hourly.nc"
            ),
            NLDAS_HOURLY_DIR=camels_us_hourly_root_dir.joinpath(
                "nldas_hourly_csv", "nldas_hourly"
            ),
            USGS_HOURLY_STREAMFLOW_DIR=camels_us_hourly_root_dir.joinpath(
                "usgs_streamflow_csv", "usgs_streamflow"
            ),
            CAMELS_GAUGE_FILE=camels_us_gague_file,
            CAMELS_ATTR_DIR=camels_us_attr_dir,
            CAMELS_US_HOURLY_DOWNLOAD_URL="https://zenodo.org/records/4072701",
        )

    def get_relevant_cols(self) -> np.array:
        return np.array(
            [
                "convective_fraction",
                "longwave_radiation",
                "potential_energy",
                "potential_evaporation",
                "pressure",
                "shortwave_radiation",
                "specific_humidity",
                "temperature",
                "total_precipitation",
                "wind_u",
                "wind_v",
            ]
        )

    def get_target_cols(self) -> np.array:
        return np.array(
            [
                "qobs_mm_per_hour",
                "qobs_count",
                "qualifiers",
                "utcoffset_hours",
                "rel_deviation_from_camels",
                "qobs_CAMELS_mm_per_hour",
            ]
        )

    def read_relevant_cols(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        forcing_type="nldas",
    ) -> np.array:
        if forcing_type not in ["nldas"]:
            raise ValueError(
                f"forcing_type must be one of ['nldas'], but got {forcing_type}"
            )
        if var_lst is None:
            return None
        if any(var not in self.get_relevant_cols() for var in var_lst):
            raise ValueError(f"var_lst must all be in {self.get_relevant_cols()}")
        start_date = t_range[0]
        # the given range is left-closed and right-open
        end_date = pd.to_datetime(t_range[1]) - pd.Timedelta(hours=1)
        t_range_lst = pd.date_range(start=start_date, end=end_date, freq="H")
        forcings_data = np.empty((len(gage_id_lst), len(t_range_lst), len(var_lst)))
        for basin in gage_id_lst:
            forcing = load_hourly_us_forcings(
                self.data_source_description["NLDAS_HOURLY_DIR"], basin, forcing_type
            )
            forcings_data[gage_id_lst.index(basin), :, :] = (
                forcing[var_lst].loc[t_range_lst[0] : t_range_lst[-1]].values
            )
        return forcings_data

    def read_target_cols(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        target_cols: list = None,
        **kwargs,
    ) -> np.array:
        if target_cols is None:
            return None
        # Define the dictionary
        key_value_dict = {
            "qobs_mm_per_hour": "QObs(mm/h)",
            "qobs_count": "QObs count",
            "qualifiers": "qualifiers",
            "utcoffset_hours": "utcoffset(h)",
            "rel_deviation_from_camels": "(iv-camels)/camels",
            "qobs_CAMELS_mm_per_hour": "QObs_CAMELS(mm/h)",
        }

        # Function to get value by key
        def get_value(key):
            return key_value_dict.get(key, "Key not found")

        if any(var not in self.get_target_cols() for var in target_cols):
            raise ValueError(f"target_cols must all be in {self.get_target_cols()}")
        tgtcols4csv = []
        for var in target_cols:
            tgtcols4csv.append(get_value(var))
        start_date = t_range[0]
        # the given range is left-closed and right-open
        end_date = pd.to_datetime(t_range[1]) - pd.Timedelta(hours=1)
        t_range_lst = pd.date_range(start=start_date, end=end_date, freq="H")
        streamflows = np.empty((len(gage_id_lst), len(t_range_lst), len(target_cols)))
        for basin in gage_id_lst:
            discharge = load_hourly_us_discharge(
                self.data_source_description["USGS_HOURLY_STREAMFLOW_DIR"], basin
            )
            streamflows[gage_id_lst.index(basin), :, :] = (
                discharge[tgtcols4csv].loc[t_range_lst[0] : t_range_lst[-1]].values
            )
        return streamflows

    def cache_xrdataset(self):
        """We save it as zarr format as it is more efficient than netcdf according to:
        https://gallery.pangeo.io/repos/earthcube2020/ec20_abernathey_etal/cloud_storage.html#Zarr-is-about-10x-faster-than-NetCDF-in-Cloud-Object-Storage

        Returns
        -------
        _type_
            _description_
        """
        ds = load_hourly_us_netcdf(
            self.data_source_description["CAMELS_US_HOURLY_TS_NC"]
        )
        ds.to_zarr(
            self.data_source_description["CAMELS_US_HOURLY_TS_NC"][:-3] + "zarr",
            mode="w",
            consolidated=True,
        )

    def read_ts_xrdataset(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs,
    ):
        if var_lst is None:
            return None
        # Define paths for Zarr and NetCDF
        zarr_path = self.data_source_description["CAMELS_US_HOURLY_TS_NC"][:-3] + "zarr"
        nc_path = self.data_source_description["CAMELS_US_HOURLY_TS_NC"]

        # Check if Zarr dataset exists, else use NetCDF
        if os.path.exists(zarr_path):
            LOGGER.info("Loading data from Zarr format. This may take a while...")
            ts = xr.open_zarr(zarr_path)
        else:
            LOGGER.info("Loading data from NetCDF format. This may take a while...")
            ts = xr.open_dataset(nc_path, chunks={"time": "auto"})

        LOGGER.info("Finish loading data.")

        if any(var not in ts.variables for var in var_lst):
            raise ValueError(f"var_lst must all be in {list(ts.variables)}")

        return ts[var_lst].sel(basin=gage_id_lst, time=slice(t_range[0], t_range[1]))


# the following code is modified from https://github.com/neuralhydrology/neuralhydrology
def load_hourly_us_forcings(
    data_dir: Path, basin: str, forcing_type: str
) -> pd.DataFrame:
    """Load the hourly forcing data for a basin of the CAMELS US data set.

    The hourly forcings are not included in the original data set by Newman et al. (2017).

    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS US forcing csv files directory.
    basin : str
        8-digit USGS identifier of the basin.
    forcing_type : str
        Must match the folder names in the 'hourly' directory. E.g. 'nldas'

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame, containing the forcing data.
    """
    file_path = data_dir.joinpath(f"{basin}_hourly_{forcing_type}.csv")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No forcing file for Basin {basin}")

    return pd.read_csv(file_path, index_col=["date"], parse_dates=["date"])


def load_hourly_us_discharge(data_dir: Path, basin: str) -> pd.DataFrame:
    """Load the hourly discharge data for a basin of the CAMELS US data set.

    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS US streamflow csv files directory.
    basin : str
        8-digit USGS identifier of the basin.

    Returns
    -------
    pd.Series
        Time-index Series of the discharge values (mm/hour)
    """
    file_path = data_dir.joinpath(f"{basin}-usgs-hourly.csv")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No streamflow file for Basin {basin}")

    return pd.read_csv(file_path, index_col=["date"], parse_dates=["date"])


def load_hourly_us_stage(data_dir: Path, basin: str) -> pd.Series:
    """Load the hourly stage data for a basin of the CAMELS US data set.
    TODO: NOT USED YET

    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS US directory. This folder must contain a folder called 'hourly' with a subdirectory
        'usgs_stage' which contains the stage files (.csv) for each basin. File names must contain the 8-digit basin id.
    basin : str
        8-digit USGS identifier of the basin.

    Returns
    -------
    pd.Series
        Time-index Series of the stage values (m)
    """
    stage_path = data_dir / "hourly" / "usgs_stage"
    files = list(stage_path.glob("**/*_utc.csv"))
    file_path = [f for f in files if basin in f.stem]
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f"No file for Basin {basin} at {stage_path}")

    df = pd.read_csv(
        file_path,
        sep=",",
        index_col=["datetime"],
        parse_dates=["datetime"],
        usecols=["datetime", "gauge_height_ft"],
    )
    df = df.resample("H").mean()
    df["gauge_height_m"] = df["gauge_height_ft"] * 0.3048

    return df["gauge_height_m"]


@Timer(name="decorator")
def load_hourly_us_netcdf(netcdf_path: Path) -> xr.Dataset:
    """Load hourly forcing and discharge data from preprocessed netCDF file.

    Parameters
    ----------
    netcdf_path : Path
        Name of the time series nc file. must be 'usgs-streamflow-nldas_hourly.nc'

    Returns
    -------
    xarray.Dataset
        Dataset containing the combined discharge and forcing data of all basins (as stored in the netCDF)
    """
    if not netcdf_path.is_file():
        raise FileNotFoundError(
            f"No NetCDF file for hourly streamflow and forcings at {netcdf_path}."
        )
    # It is very slow when directly loading it, so we use dask
    return xr.open_dataset(netcdf_path, chunks={"time": "auto", "basin": "auto"})
