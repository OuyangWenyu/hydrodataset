from hydrodataset.camelsh import Camelsh
import xarray as xr
from hydrodataset import CACHE_DIR, SETTING
import numpy as np

data_path = SETTING["local_data_path"]["root"]
import pandas as pd


# "Test whether read_attr_xrdataset() correctly reads .nc files and returns a list of watershed ID strings."
def test_read_attr_xrdataset():
    ds = Camelsh(data_path)
    result_1 = ds.read_attr_xrdataset(gage_id_lst='01011000', var_lst=["p_mean"])[
        'p_mean'
    ].values
    csv_path = r"D:\data\CAMELSH\attributes\attributes_nldas2_climate.csv"
    df = pd.read_csv(csv_path)
    # 获取STAID为01011000的p_mean值
    result_2 = df[df['STAID'] == 1011000]['p_mean'].values[0]
    assert result_1 == result_2


# "Test whether read_ts_xrdataset() correctly reads .nc files and returns a list of watershed ID strings."
def test_read_timeseries_xrdataset():
    ds = Camelsh(data_path)
    ts_data = ds.read_ts_xrdataset(
        gage_id_lst=['01011000'],
        var_lst=['pet_mm'],
        t_range=['1980-01-01', '1980-01-01'],
    )
    station_data = ts_data['pet_mm']
    result_1 = station_data.values.flatten()
    file_path = "D:/data/CAMELSH/timeseries/Data/CAMELSH/timeseries/01011000.nc"
    ds = xr.open_dataset(file_path)
    pet_data = ds['PotEvap']
    result_2 = pet_data.isel(DateTime=slice(0, 24)).values
    values_match = np.array_equal(result_1, result_2)
    assert values_match
