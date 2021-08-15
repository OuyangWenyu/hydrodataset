"""
Process NLDAS forcing data for basins
"""
import fnmatch
import os
import numpy as np
import pandas as pd


def trans_daily_nldas_to_camels_format(nldas_dir, output_dir, gage_dict, region, year):
    """
    Transform daily forcing data of NLDAS downloaded from GEE to the format in CAMELS.

    The GEE code used to generate the original data can be seen here:
    https://code.earthengine.google.com/6770e9bd20af10f70838fc04148cc122
    If you can read Chinese, and prefer Python code, you can see here:
    https://github.com/OuyangWenyu/hydroGIS/blob/master/GEE/4-geepy-gallery.ipynb

    NLDAS's forcing data is hourly, and we use GEE to get the daily values for basins.

    Parameters
    ----------
    nldas_dir
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
    avg_dataset = ['gage_id', "time_start", "temperature_mean", "specific_humidity_mean", "pressure_mean",
                   "wind_u_mean", "wind_v_mean", "longwave_radiation_mean", "convective_fraction_mean",
                   "shortwave_radiation_mean"]
    sum_dataset = ['gage_id', "time_start", "potential_energy_sum", "potential_evaporation_sum",
                   "total_precipitation_sum"]
    camels_format_index = ['Year', 'Mnth', 'Day', 'Hr', 'temperature(C)', 'specific_humidity(kg/kg)', 'pressure(Pa)',
                           'wind_u(m/s)', 'wind_v(m/s)', 'longwave_radiation(W/m^2)', 'convective_fraction(-)',
                           'shortwave_radiation(W/m^2)', 'potential_energy(J/kg)', 'potential_evaporation(kg/m^2)',
                           'total_precipitation(kg/m^2)']

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
    for f_name in os.listdir(nldas_dir):
        if fnmatch.fnmatch(f_name, 'nldas_' + region + '_avg_mean_' + str(year) + '.csv'):
            avg_data_file = os.path.join(nldas_dir, f_name)
        if fnmatch.fnmatch(f_name, 'nldas_' + region + '_sum_mean_' + str(year) + '.csv'):
            sum_data_file = os.path.join(nldas_dir, f_name)

    avg_data_temp = pd.read_csv(avg_data_file, sep=',', dtype={avg_dataset[0]: str})
    sum_data_temp = pd.read_csv(sum_data_file, sep=',', dtype={sum_dataset[0]: str})
    for i_basin in range(len(gage_dict[gage_id_key])):
        avg_basin_data = avg_data_temp[
            avg_data_temp[avg_dataset[0]].values.astype(int) == int(gage_dict[gage_id_key][i_basin])]
        sum_basin_data = sum_data_temp[
            sum_data_temp[avg_dataset[0]].values.astype(int) == int(gage_dict[gage_id_key][i_basin])]
        if avg_basin_data.shape[0] == 0 or sum_basin_data.shape[0] == 0:
            raise ArithmeticError("chosen basins' number is zero")
        # get Year,Month,Day,Hour info
        csv_date = pd.to_datetime(avg_basin_data[avg_dataset[1]])
        # the hour is set to 12, as 12 is the average hour of a day
        year_month_day_hour = pd.DataFrame([[dt.year, dt.month, dt.day, 12] for dt in csv_date],
                                           columns=camels_format_index[0:4])
        avg_data_df = pd.DataFrame(avg_basin_data.iloc[:, 2:].values, columns=camels_format_index[4:-3])
        sum_data_df = pd.DataFrame(sum_basin_data.iloc[:, 2:].values, columns=camels_format_index[-3:])
        # concat
        new_data_df = pd.concat([year_month_day_hour, avg_data_df, sum_data_df], axis=1)
        # output the result
        huc_id = gage_dict[huc02_key][i_basin]
        output_huc_dir = os.path.join(output_dir, huc_id)
        if not os.path.isdir(output_huc_dir):
            os.makedirs(output_huc_dir)
        output_file = os.path.join(output_huc_dir, gage_dict[gage_id_key][i_basin] + '_lump_nldas_forcing_leap.txt')
        print("output forcing data of", gage_dict[gage_id_key][i_basin], "year", str(year))
        if os.path.isfile(output_file):
            data_old = pd.read_csv(output_file, sep=' ')
            years = np.unique(data_old[camels_format_index[0]].values)
            if year in years:
                continue
            else:
                os.remove(output_file)
                new_data_df = pd.concat([data_old, new_data_df]).sort_values(by=camels_format_index[0:3])
        new_data_df.to_csv(output_file, header=True, index=False, sep=' ', float_format='%.2f')
