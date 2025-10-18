import os
from hydrodataset import CACHE_DIR
from hydrodataset.camels import Camels
import xarray as xr


class Camels4EALSTM(Camels):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def read_ts_xrdataset(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs,
    ):
        if var_lst is None:
            return None
        camels_tsnc = CACHE_DIR.joinpath("camelsus_timeseries.nc")
        if not os.path.isfile(camels_tsnc):
            self.cache_xrdataset()
        ts = xr.open_dataset(camels_tsnc)
        all_vars = ts.data_vars
        if any(var not in ts.variables for var in var_lst):
            raise ValueError(f"var_lst must all be in {all_vars}")

        orig_attrs = ts['streamflow'].attrs
        data_mean = ts[var_lst].sel(basin=gage_id_lst, time=slice(t_range[0], t_range[1])).mean(dim='time')

        if 'streamflow' in data_mean:
            data_mean['streamflow'].attrs = orig_attrs
        return data_mean
