"""
Process PMLV2 ET data for basins
"""
import fnmatch
import os
import numpy as np
import pandas as pd
import datetime


def trans_8day_pmlv2_to_camels_format(pmlv2_dir, output_dir, gage_dict, region, year):
    """
    Transform 8-day PMLV2 data downloaded from GEE to the format in CAMELS.

    If you can read Chinese, and prefer Python code, you can see here:
    https://github.com/wangmengyun1998/hydroGIS/blob/master/GEE/4-geepy-gallery.ipynb

    Parameters
    ----------
    pmlv2_dir
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
    pmlv2_dataset = ["hru_id", "system:time_start", "GPP", "Ec", "Es", "Ei", "ET_water"]
    camels_format_index = ["Year", "Mnth", "Day", "Hr", "GPP(gC/m2/d)", "Ec(mm/d)", "Es(mm/d)", "Ei(mm/d)",
                           "ET_water(mm/d)"]

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
    for f_name in os.listdir(pmlv2_dir):
        if fnmatch.fnmatch(f_name, 'PML_V2_' + region + '_mean_' + str(year) + '*.csv'):
            pmlv2_data_file = os.path.join(pmlv2_dir, f_name)

    data_temp = pd.read_csv(pmlv2_data_file, sep=',', dtype={pmlv2_dataset[0]: str})
    for i_basin in range(len(gage_dict[gage_id_key])):
        basin_data = data_temp[
            data_temp[pmlv2_dataset[0]].values.astype(int) == int(gage_dict[gage_id_key][i_basin])]
        if basin_data.shape[0] == 0:
            raise ArithmeticError("chosen basins' number is zero")
        # get Year,Month,Day,Hour info
        # if system:time_start is millisecond
        tmp_times = [datetime.datetime.fromtimestamp(tmp) for tmp in basin_data[pmlv2_dataset[1]].values/1000]
        csv_date = pd.to_datetime(tmp_times)
        # csv_date = pd.to_datetime(basin_data[pmlv2_dataset[1]])
        # the hour is set to 12, as 12 is the average hour of a day
        year_month_day_hour = pd.DataFrame([[dt.year, dt.month, dt.day, 12] for dt in csv_date],
                                           columns=camels_format_index[0:4])
        data_df = pd.DataFrame(basin_data.iloc[:, 2:].values, columns=camels_format_index[4:])
        # concat
        new_data_df = pd.concat([year_month_day_hour, data_df], axis=1)
        # output the result
        huc_id = gage_dict[huc02_key][i_basin]
        output_huc_dir = os.path.join(output_dir, huc_id)
        if not os.path.isdir(output_huc_dir):
            os.makedirs(output_huc_dir)
        output_file = os.path.join(output_huc_dir, gage_dict[gage_id_key][i_basin] + '_lump_pmlv2_et.txt')
        print("output pmlv2 et data of", gage_dict[gage_id_key][i_basin], "year", str(year))
        if os.path.isfile(output_file):
            data_old = pd.read_csv(output_file, sep=' ')
            years = np.unique(data_old[camels_format_index[0]].values)
            if year in years:
                continue
            else:
                os.remove(output_file)
                new_data_df = pd.concat([data_old, new_data_df]).sort_values(by=camels_format_index[0:3])
        new_data_df.to_csv(output_file, header=True, index=False, sep=' ', float_format='%.4f')
