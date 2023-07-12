"""
Author: Wenyu Ouyang
Date: 2023-01-02 22:23:24
LastEditTime: 2023-01-02 22:59:26
LastEditors: Wenyu Ouyang
Description: read the Global Runoff Data Centre (GRDC) data
FilePath: \hydrodataset\hydrodataset\grdc.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
# Global Runoff Data Centre module from ewatercycle: https://github.com/eWaterCycle/ewatercycle/blob/main/src/ewatercycle/observation/grdc.py
import datetime
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from dateutil.parser import parse
import pandas as pd
from hydrodataset import ROOT_DIR

logger = logging.getLogger(__name__)

MetaDataType = Dict[str, Union[str, int, float]]

# TODO rename to to_utcdatetime
def get_time(time_iso: str) -> datetime.datetime:
    """Return a datetime in UTC.
    Convert a date string in ISO format to a datetime
    and check if it is in UTC.
    """
    time = parse(time_iso)
    if time.tzname() != "UTC":
        raise ValueError(
            "The time is not in UTC. The ISO format for a UTC time "
            "is 'YYYY-MM-DDTHH:MM:SSZ'"
        )
    return time


def to_absolute_path(
    input_path: str,
    parent: Optional[Path] = None,
    must_exist: bool = False,
    must_be_in_parent=True,
) -> Path:
    """Parse input string as :py:class:`pathlib.Path` object.
    Args:
        input_path: Input string path that can be a relative or absolute path.
        parent: Optional parent path of the input path
        must_exist: Optional argument to check if the input path exists.
        must_be_in_parent: Optional argument to check if the input path is
            subpath of parent path
    Returns:
        The input path that is an absolute path and a :py:class:`pathlib.Path` object.
    """
    pathlike = Path(input_path)
    if parent:
        pathlike = parent.joinpath(pathlike)
        if must_be_in_parent:
            try:
                pathlike.relative_to(parent)
            except ValueError as e:
                raise ValueError(
                    f"Input path {input_path} is not a subpath of parent {parent}"
                ) from e

    return pathlike.expanduser().resolve(strict=must_exist)


def get_grdc_data(
    station_id: str,
    start_time: str,
    end_time: str,
    parameter: str = "Q",
    data_home: Optional[str] = None,
    column: str = "streamflow",
) -> Tuple[pd.core.frame.DataFrame, MetaDataType]:
    """Get river discharge data from Global Runoff Data Centre (GRDC).

    Requires the GRDC daily data files in a local directory. The GRDC daily data
    files can be ordered at
    https://www.bafg.de/GRDC/EN/02_srvcs/21_tmsrs/riverdischarge_node.html

    Args:
        station_id: The station id to get. The station id can be found in the
            catalogues at
            https://www.bafg.de/GRDC/EN/02_srvcs/21_tmsrs/212_prjctlgs/project_catalogue_node.html
        start_time: Start time of model in UTC and ISO format string e.g.
            'YYYY-MM-DDTHH:MM:SSZ'.
        end_time: End time of model in  UTC and ISO format string e.g.
            'YYYY-MM-DDTHH:MM:SSZ'.
        parameter: optional. The parameter code to get, e.g. ('Q') discharge,
            cubic meters per second.
        data_home : optional. The directory where the daily grdc data is
            located. If left out will use the grdc_location in the eWaterCycle
            configuration file.
        column: optional. Name of column in dataframe. Default: "streamflow".

    Returns:
        grdc data in a dataframe and metadata.

    Examples:
        .. code-block:: python

            from ewatercycle.observation.grdc import get_grdc_data

            df, meta = get_grdc_data('6335020',
                                    '2000-01-01T00:00Z',
                                    '2001-01-01T00:00Z')
            df.describe()
                     streamflow
            count   4382.000000
            mean    2328.992469
            std	    1190.181058
            min	     881.000000
            25%	    1550.000000
            50%	    2000.000000
            75%	    2730.000000
            max	   11300.000000

            meta
            {'grdc_file_name': '/home/myusername/git/eWaterCycle/ewatercycle/6335020_Q_Day.Cmd.txt',
            'id_from_grdc': 6335020,
            'file_generation_date': '2019-03-27',
            'river_name': 'RHINE RIVER',
            'station_name': 'REES',
            'country_code': 'DE',
            'grdc_latitude_in_arc_degree': 51.756918,
            'grdc_longitude_in_arc_degree': 6.395395,
            'grdc_catchment_area_in_km2': 159300.0,
            'altitude_masl': 8.0,
            'dataSetContent': 'MEAN DAILY DISCHARGE (Q)',
            'units': 'm³/s',
            'time_series': '1814-11 - 2016-12',
            'no_of_years': 203,
            'last_update': '2018-05-24',
            'nrMeasurements': 'NA',
            'UserStartTime': '2000-01-01T00:00Z',
            'UserEndTime': '2001-01-01T00:00Z',
            'nrMissingData': 0}
    """  # noqa: E501
    if data_home:
        data_path = to_absolute_path(data_home)
    else:
        raise ValueError(
            "Provide the grdc path using `data_home` argument"
            "or using `grdc_location` in ewatercycle configuration file."
        )

    if not data_path.exists():
        raise ValueError(f"The grdc directory {data_path} does not exist!")

    # Read the raw data
    raw_file = data_path / f"{station_id}_{parameter}_Day.Cmd.txt"
    if not raw_file.exists():
        raise ValueError(f"The grdc file {raw_file} does not exist!")

    # Convert the raw data to an xarray
    metadata, df = _grdc_read(
        raw_file,
        start=get_time(start_time).date(),
        end=get_time(end_time).date(),
        column=column,
    )

    # Add start/end_time to metadata
    metadata["UserStartTime"] = start_time
    metadata["UserEndTime"] = end_time

    # Add number of missing data to metadata
    metadata["nrMissingData"] = _count_missing_data(df, column)

    # Shpw info about data
    _log_metadata(metadata)

    return df, metadata


def _grdc_read(grdc_station_path, start, end, column):
    with grdc_station_path.open("r", encoding="cp1252", errors="ignore") as file:
        data = file.read()

    metadata = _grdc_metadata_reader(grdc_station_path, data)

    all_lines = data.split("\n")
    header = 0
    for i, line in enumerate(all_lines):
        if line.startswith("# DATA"):
            header = i + 1
            break

    # Import GRDC data into dataframe and modify dataframe format
    grdc_data = pd.read_csv(
        grdc_station_path,
        encoding="cp1252",
        skiprows=header,
        delimiter=";",
        parse_dates=["YYYY-MM-DD"],
        na_values="-999",
    )
    grdc_station_df = pd.DataFrame(
        {column: grdc_data[" Value"].array},
        index=grdc_data["YYYY-MM-DD"].array,
    )
    grdc_station_df.index.rename("time", inplace=True)

    # Select GRDC station data that matches the forecast results Date
    grdc_station_select = grdc_station_df.loc[start:end]

    return metadata, grdc_station_select


def _grdc_metadata_reader(grdc_station_path, all_lines):
    # Initiating a dictionary that will contain all GRDC attributes.
    # This function is based on earlier work by Rolf Hut.
    # https://github.com/RolfHut/GRDC2NetCDF/blob/master/GRDC2NetCDF.py
    # DOI: 10.5281/zenodo.19695
    # that function was based on earlier work by Edwin Sutanudjaja
    # from Utrecht University.
    # https://github.com/edwinkost/discharge_analysis_IWMI
    # Modified by Susan Branchett

    # initiating a dictionary that will contain all GRDC attributes:
    attribute_grdc = {}

    # split the content of the file into several lines
    all_lines = all_lines.replace("\r", "")
    all_lines = all_lines.split("\n")

    # get grdc ids (from files) and check their consistency with their
    # file names
    id_from_file_name = int(
        os.path.basename(grdc_station_path).split(".")[0].split("_")[0]
    )
    id_from_grdc = None
    if id_from_file_name == int(all_lines[8].split(":")[1].strip()):
        id_from_grdc = int(all_lines[8].split(":")[1].strip())
    else:
        print(
            "GRDC station "
            + str(id_from_file_name)
            + " ("
            + str(grdc_station_path)
            + ") is NOT used."
        )

    if id_from_grdc is not None:

        attribute_grdc["grdc_file_name"] = str(grdc_station_path)
        attribute_grdc["id_from_grdc"] = id_from_grdc

        try:
            attribute_grdc["file_generation_date"] = str(
                all_lines[6].split(":")[1].strip()
            )
        except (IndexError, ValueError):
            attribute_grdc["file_generation_date"] = "NA"

        try:
            attribute_grdc["river_name"] = str(all_lines[9].split(":")[1].strip())
        except (IndexError, ValueError):
            attribute_grdc["river_name"] = "NA"

        try:
            attribute_grdc["station_name"] = str(all_lines[10].split(":")[1].strip())
        except (IndexError, ValueError):
            attribute_grdc["station_name"] = "NA"

        try:
            attribute_grdc["country_code"] = str(all_lines[11].split(":")[1].strip())
        except (IndexError, ValueError):
            attribute_grdc["country_code"] = "NA"

        try:
            attribute_grdc["grdc_latitude_in_arc_degree"] = float(
                all_lines[12].split(":")[1].strip()
            )
        except (IndexError, ValueError):
            attribute_grdc["grdc_latitude_in_arc_degree"] = "NA"

        try:
            attribute_grdc["grdc_longitude_in_arc_degree"] = float(
                all_lines[13].split(":")[1].strip()
            )
        except (IndexError, ValueError):
            attribute_grdc["grdc_longitude_in_arc_degree"] = "NA"

        try:
            attribute_grdc["grdc_catchment_area_in_km2"] = float(
                all_lines[14].split(":")[1].strip()
            )
            if attribute_grdc["grdc_catchment_area_in_km2"] <= 0.0:
                attribute_grdc["grdc_catchment_area_in_km2"] = "NA"
        except (IndexError, ValueError):
            attribute_grdc["grdc_catchment_area_in_km2"] = "NA"

        try:
            attribute_grdc["altitude_masl"] = float(all_lines[15].split(":")[1].strip())
        except (IndexError, ValueError):
            attribute_grdc["altitude_masl"] = "NA"

        try:
            attribute_grdc["dataSetContent"] = str(all_lines[20].split(":")[1].strip())
        except (IndexError, ValueError):
            attribute_grdc["dataSetContent"] = "NA"

        try:
            attribute_grdc["units"] = str(all_lines[22].split(":")[1].strip())
        except (IndexError, ValueError):
            attribute_grdc["units"] = "NA"

        try:
            attribute_grdc["time_series"] = str(all_lines[23].split(":")[1].strip())
        except (IndexError, ValueError):
            attribute_grdc["time_series"] = "NA"

        try:
            attribute_grdc["no_of_years"] = int(all_lines[24].split(":")[1].strip())
        except (IndexError, ValueError):
            attribute_grdc["no_of_years"] = "NA"

        try:
            attribute_grdc["last_update"] = str(all_lines[25].split(":")[1].strip())
        except (IndexError, ValueError):
            attribute_grdc["last_update"] = "NA"

        try:
            attribute_grdc["nrMeasurements"] = int(
                str(all_lines[33].split(":")[1].strip())
            )
        except (IndexError, ValueError):
            attribute_grdc["nrMeasurements"] = "NA"

    return attribute_grdc


def _count_missing_data(df, column):
    """Return number of missing data."""
    return df[column].isna().sum()


def _log_metadata(metadata):
    """Print some information about data."""
    coords = (
        metadata["grdc_latitude_in_arc_degree"],
        metadata["grdc_longitude_in_arc_degree"],
    )
    message = (
        f"GRDC station {metadata['id_from_grdc']} is selected. "
        f"The river name is: {metadata['river_name']}."
        f"The coordinates are: {coords}."
        f"The catchment area in km2 is: {metadata['grdc_catchment_area_in_km2']}. "
        f"There are {metadata['nrMissingData']} missing values during "
        f"{metadata['UserStartTime']}_{metadata['UserEndTime']} at this station. "
        f"See the metadata for more information."
    )
    logger.info("%s", message)


if __name__ == "__main__":
    df, metadata = get_grdc_data(
        station_id="2181200",
        start_time="1980-01-01T00:00Z",
        end_time="2001-01-01T00:00Z",
        data_home=Path(ROOT_DIR, "grdc"),
    )
    df.to_csv(Path(ROOT_DIR, "grdc", "2181200.csv"))
