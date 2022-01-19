"""
Process MODIS16A2V105 ET data for basins
"""
import fnmatch
import logging
import os
import numpy as np
import pandas as pd


def trans_month_nex_dcp30to_camels_format(
    nexdcp30_dir, output_dir, gage_dict, region, year
):
    """
    Transform NEX-DCP30 data downloaded from GEE to the format in CAMELS.
    https://code.earthengine.google.com/5edfca6263bea36f5c093fc6b80a68aa

    Parameters
    ----------
    nexdcp30_dir
        the original data's directory
    output_dir
        the transformed data's directory
    gage_dict
        a dict containing gage's ids and the correspond HUC02 ids
    region
        we named the file downloaded from GEE as daymet_<region>_mean_<year>.csv,
        because we use GEE code to generate data for each year for each shape file (region) containing some basins.
        For example, if we use the basins' shpfile in CAMELS, the region is "camels".
    year
        we use GEE code to generate data for each year, so each year for each region has one data file.
    Returns
    -------
    None

    """
    # you can add features or delete features, or change the order, which depends on your txt content
    nexdcp30_dataset = [
        "gage_id",
        "time_start",
        "pr_mean",
        "pr_quartile25",
        "pr_median",
        "pr_quartile75",
        "tasmin_mean",
        "tasmin_quartile25",
        "tasmin_median",
        "tasmin_quartile75",
        "tasmax_mean",
        "tasmax_quartile25",
        "tasmax_median",
        "tasmax_quartile75",
    ]
    camels_format_index = [
        "Year",
        "Mnth",
        "Day",
        "Hr",
        "pr_mean(kg/(m^2*s))",
        "pr_quartile25(kg/(m^2*s))",
        "pr_median(kg/(m^2*s))",
        "pr_quartile75(kg/(m^2*s))",
        "tasmin_mean(K)",
        "tasmin_quartile25(K)",
        "tasmin_median(K)",
        "tasmin_quartile75(K)",
        "tasmax_mean(K)",
        "tasmax_quartile25(K)",
        "tasmax_median(K)",
        "tasmax_quartile75(K)",
    ]

    if "STAID" in gage_dict.keys():
        gage_id_key = "STAID"
    elif "gauge_id" in gage_dict.keys():
        gage_id_key = "gauge_id"
    elif "gage_id" in gage_dict.keys():
        gage_id_key = "gage_id"
    else:
        raise NotImplementedError("No such gage id name")

    if "HUC02" in gage_dict.keys():
        huc02_key = "HUC02"
    elif "huc_02" in gage_dict.keys():
        huc02_key = "huc_02"
    else:
        raise NotImplementedError("No such huc02 id")

    # because this function only work for one year and one region, it's better to chose avg and sum files at first
    for f_name in os.listdir(nexdcp30_dir):
        if fnmatch.fnmatch(
            f_name, "nex_dcp30_" + region + "_mean_" + str(year) + "*.csv"
        ):
            nexdcp30_data_file = os.path.join(nexdcp30_dir, f_name)

    data_temp = pd.read_csv(
        nexdcp30_data_file, sep=",", dtype={nexdcp30_data_file[0]: str}
    )
    for i_basin in range(len(gage_dict[gage_id_key])):
        basin_data = data_temp[
            data_temp[nexdcp30_dataset[0]].values.astype(int)
            == int(gage_dict[gage_id_key][i_basin])
        ]
        if basin_data.shape[0] == 0:
            logging.warning("No data for basin %s", gage_dict[gage_id_key][i_basin])
            continue
        # get Year,Month,Day,Hour info
        csv_date = pd.to_datetime(basin_data[nexdcp30_dataset[1]])
        # the hour is set to 12, as 12 is the average hour of a day
        year_month_day_hour = pd.DataFrame(
            [[dt.year, dt.month, dt.day, 12] for dt in csv_date],
            columns=camels_format_index[0:4],
        )
        data_df = pd.DataFrame(
            basin_data.iloc[:, 2:].values, columns=camels_format_index[4:]
        )
        # concat
        new_data_df = pd.concat([year_month_day_hour, data_df], axis=1)
        # output the result
        huc_id = gage_dict[huc02_key][i_basin]
        output_huc_dir = os.path.join(output_dir, huc_id)
        if not os.path.isdir(output_huc_dir):
            os.makedirs(output_huc_dir)
        if basin_data.shape[0] > 12:
            # months>12 means different RCP projections; ==12 means "historical"
            if basin_data.shape[0] != 48:
                raise RuntimeError(
                    "Maybe you lose some RCP projections: 'rcp26', 'rcp45', 'rcp60', 'rcp85'"
                )
            # TODO: GEE code don't explicitly point out which RCP projection, so we directly specify them
            rcp26 = new_data_df[0:12]
            rcp45 = new_data_df[12:24]
            rcp60 = new_data_df[24:36]
            rcp85 = new_data_df[36:48]
            new_data_df_lst = [rcp26, rcp45, rcp60, rcp85]
            out_file_name_lst = ["rcp26", "rcp45", "rcp60", "rcp85"]
        else:
            new_data_df_lst = [new_data_df]
            out_file_name_lst = ["historical"]
        for i in range(len(new_data_df_lst)):
            _data_df = new_data_df_lst[i]
            output_file = os.path.join(
                output_huc_dir,
                gage_dict[gage_id_key][i_basin]
                + "_lump_nexdcp30_"
                + out_file_name_lst[i]
                + ".txt",
            )
            print(
                "output nexdcp30 " + out_file_name_lst[i] + " of",
                gage_dict[gage_id_key][i_basin],
                "year",
                str(year),
            )
            if os.path.isfile(output_file):
                data_old = pd.read_csv(output_file, sep=" ")
                years = np.unique(data_old[camels_format_index[0]].values)
                if year in years:
                    continue
                else:
                    os.remove(output_file)
                    _data_df = pd.concat([data_old, _data_df]).sort_values(
                        by=camels_format_index[0:3]
                    )
            _data_df.to_csv(output_file, header=True, index=False, sep=" ")
