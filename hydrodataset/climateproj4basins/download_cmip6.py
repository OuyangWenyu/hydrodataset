import logging
import os
from black import out

import numpy as np
import wget
from tqdm import tqdm


class NexGddpCmip6:
    def __init__(self):
        self.gcms = [
            "UKESM1-0-LL",
            "TaiESM1",
            "NorESM2-MM",
            "NorESM2-LM",
            "NESM3",
            "MRI-ESM2-0",
            "MPI-ESM1-2-LR",
            "MPI-ESM1-2-HR",
            "MIROC6",
            "MIROC-ES2L",
            "KIOST-ESM",
            "KACE-1-0-G",
            "IPSL-CM6A-LR",
            "INM-CM5-0",
            "INM-CM4-8",
            "IITM-ESM",
            "HadGEM3-GC31-MM",
            "HadGEM3-GC31-LL",
            "GISS-E2-1-G",
            "GFDL-ESM4",
            "GFDL-CM4_gr2",
            "GFDL-CM4",
            "FGOALS-g3",
            "EC-Earth3-Veg-LR",
            "EC-Earth3",
            "CanESM5",
            "CNRM-ESM2-1",
            "CNRM-CM6-1",
            "CMCC-ESM2",
            "CMCC-CM2-SR5",
            "CESM2-WACCM",
            "CESM2",
            "BCC-CSM2-MR",
            "ACCESS-ESM1-5",
            "ACCESS-CM2",
        ]
        self.scenarios = ["ssp585", "ssp245", "historical"]
        self.cases = ["r1i1p1f1", "r1i1p1f2", "r1i1p1f3"]
        self.future_year_range = [2015, 2100]
        self.history_year_range = [1950, 2014]
        self.vars_type = [
            "hurs",
            "huss",
            "pr",
            "rlds",
            "rsds",
            "sfcWind",
            "tas",
            "tasmax",
            "tasmin",
        ]
        self.url_pattern = (
            "https://ds.nccs.nasa.gov/thredds2/ncss/AMES/NEX/GDDP-CMIP6/{gcm}/{scenario}/"
            "{case}/{var}/{var}_day_{gcm}_{scenario}_{case}_gn_{year}.nc?"
            "var={var}&north={north}&west={west}&east={east}&south={south}&"
            "disableProjSubset=on&horizStride=1&time_start={year}-01-01T12%3A00%3A00Z&"
            "time_end={year}-12-31T12%3A00%3A00Z&timeStride=1"
        )

    def download_all_nex_gddp_cmip6_for_a_region(
        self, north, east, south, west, save_dir
    ):
        """
        Download all NEX-GDDP-CMIP6 for a region

        For example, for CONUS: north=51, west=234, east=294, south=23

        Parameters
        ----------
        north
            north latitude
        east
            east longitude
        south
            south latitude
        west
            west longitude
        save_dir
            where we save downloaded data

        Returns
        -------
        None
        """
        for gcm in tqdm(self.gcms):
            for sce in tqdm(self.scenarios, leave=False):
                self.download_one_gcm_scenario_nex_gddp_cmip6_for_a_region(
                    gcm, sce, north, east, south, west, save_dir
                )

    def download_one_gcm_scenario_nex_gddp_cmip6_for_a_region(
        self, gcm, sce, north, east, south, west, save_dir
    ):
        """
        Download all NEX-GDDP-CMIP6 for a region

        For example, for CONUS: north=51, west=234, east=294, south=23

        Parameters
        ----------
        gcm
            which gcm
        sce
            which scenario -- "ssp585", "ssp245", "historical"
        north
            north latitude
        east
            east longitude
        south
            south latitude
        west
            west longitude
        save_dir
            where we save downloaded data

        Returns
        -------
        None
        """
        if sce == "historical":
            year_range = self.history_year_range
        else:
            year_range = self.future_year_range
        for year in tqdm(np.arange(year_range[0], year_range[1] + 1)):
            for var in tqdm(self.vars_type, leave=False):
                self.download_one_nex_gddp_cmip6_file_for_a_region(
                    gcm, sce, year, var, north, east, south, west, save_dir
                )

    def download_one_nex_gddp_cmip6_file_for_a_region(
        self, gcm, sce, year, var, north, east, south, west, save_dir
    ):
        """
        Download all NEX-GDDP-CMIP6 for a region

        For example, for CONUS: north=51, west=234, east=294, south=23

        Parameters
        ----------
        gcm
            which gcm
        sce
            which scenario -- "ssp585", "ssp245", "historical"
        year
            one year
        var
            one variable
        north
            north latitude
        east
            east longitude
        south
            south latitude
        west
            west longitude
        save_dir
            where we save downloaded data

        Returns
        -------
        None
        """
        if gcm in [
            "CNRM-CM6-1",
            "CNRM-ESM2-1",
            "GISS-E2-1-G",
            "MIROC-ES2L",
            "UKESM1-0-LL",
        ]:
            case = self.cases[1]
        elif gcm in ["HadGEM3-GC31-MM", "HadGEM3-GC31-LL"]:
            case = self.cases[-1]
        else:
            case = self.cases[0]
        if gcm == "HadGEM3-GC31-MM" and sce == "ssp245":
            logging.warning("No available data for HadGEM3-GC31-MM ssp245")
            return
        if sce == "historical":
            year_range = self.history_year_range
        else:
            year_range = self.future_year_range
        if year < year_range[0] or year > year_range[1]:
            logging.warning("No available data in this year")
            return
        if (
            (gcm in ["BCC-CSM2-MR", "NESM3"] and var == "hurs")
            or (gcm in ["IPSL-CM6A-LR", "MIROC6", "NESM3"] and var == "huss")
            or (
                gcm in ["CESM2", "CESM2-WACCM", "IITM-ESM"]
                and var in ["tasmax", "tasmin"]
            )
        ):
            logging.warning("No available data for this" + gcm + " " + var)
            return
        url = self.url_pattern.format(
            gcm=gcm,
            scenario=sce,
            case=case,
            year=year,
            var=var,
            north=north,
            west=west,
            east=east,
            south=south,
        )
        file_name = "{var}_day_{gcm}_{scenario}_{case}_gn_{year}.nc".format(
            gcm=gcm, scenario=sce, case=case, year=year, var=var
        )
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        out_file = os.path.join(save_dir, file_name)
        if os.path.isfile(out_file):
            print(out_file + " has been downloaded!")
        else:
            wget.download(url, out=out_file)
