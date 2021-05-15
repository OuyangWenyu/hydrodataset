import datetime
import numpy as np


def date_to_julian(a_time):
    if type(a_time) == str:
        fmt = '%Y-%m-%d'
        dt = datetime.datetime.strptime(a_time, fmt)
    else:
        dt = a_time
    tt = dt.timetuple()
    julian_date = tt.tm_yday
    return julian_date


def t_range_to_julian(t_range):
    t_array = t_range_days(t_range)
    t_array_str = np.datetime_as_string(t_array)
    julian_dates = [date_to_julian(a_time[0:10]) for a_time in t_array_str]
    return julian_dates


def t_range_days(t_range, *, step=np.timedelta64(1, 'D')):
    sd = datetime.datetime.strptime(t_range[0], '%Y-%m-%d')
    ed = datetime.datetime.strptime(t_range[1], '%Y-%m-%d')
    t_array = np.arange(sd, ed, step)
    return t_array
