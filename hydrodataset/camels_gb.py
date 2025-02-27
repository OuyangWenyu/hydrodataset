import json
import warnings


    def cache_forcing_np_json(self):
        """
        Save all basin-forcing data in a numpy array file in the cache directory.

        Because it takes much time to read data from csv files,
        it is a good way to cache data as a numpy file to speed up the reading.
        In addition, we need a document to explain the meaning of all dimensions.

        """
        cache_npy_file = CACHE_DIR.joinpath("camels_gb_forcing.npy")
        json_file = CACHE_DIR.joinpath("camels_gb_forcing.json")
        variables = self.get_relevant_cols()
        basins = self.sites["gauge_id"].values
        t_range = ["1970-10-01", "2015-09-30"]
        times = [
            hydro_time.t2str(tmp)
            for tmp in hydro_time.t_range_days(t_range).tolist()
        ]
        data_info = collections.OrderedDict(
            {
                "dim": ["basin", "time", "variable"],
                "basin": basins.tolist(),
                "time": times,
                "variable": variables.tolist(),
            }
        )
        with open(json_file, "w") as FP:
            json.dump(data_info, FP, indent=4)
        data = self.read_relevant_cols(
            gage_id_lst=basins.tolist(),
            t_range=t_range,
            var_lst=variables.tolist(),
        )
        np.save(cache_npy_file, data)

    def cache_streamflow_np_json(self):
        """
        Save all basins' streamflow data in a numpy array file in the cache directory
        """
        cache_npy_file = CACHE_DIR.joinpath("camels_gb_streamflow.npy")
        json_file = CACHE_DIR.joinpath("camels_gb_streamflow.json")
        variables = self.get_target_cols()
        basins = self.sites["gauge_id"].values
        t_range = ["1970-10-01", "2015-09-30"]
        times = [
            hydro_time.t2str(tmp) for tmp in hydro_time.t_range_days(t_range).tolist()
        ]
        data_info = collections.OrderedDict(
            {
                "dim": ["basin", "time", "variable"],
                "basin": basins.tolist(),
                "time": times,
                "variable": variables.tolist(),
            }
        )
        with open(json_file, "w") as FP:
            json.dump(data_info, FP, indent=4)
        data = self.read_target_cols(
            gage_id_lst=basins,
            t_range=t_range,
            target_cols=variables,
        )
        np.save(cache_npy_file, data)

    def cache_attributes_xrdataset(self):
        """Convert all the attributes to a single dataframe

        Returns
        -------
        None
        """
        # NOTICE: although it seems that we don't use pint_xarray, we have to import this package
        import pint_xarray

        attr_files = self.data_source_dir.glob("CAMELS_GB_*.cvs")
        attrs = {
            f.stem.split("_")[1]: pd.read_csv(
                f, sep=";", index_col=0, dtype={"gauge_id": str}
            )
            for f in attr_files
        }

        attrs_df = pd.concat(attrs.values(), axis=1)

        # fix station names
        def fix_station_nm(station_nm):
            name = station_nm.title().rsplit(" ", 1)
            name[0] = name[0] if name[0][-1] == "," else f"{name[0]},"
            name[1] = name[1].replace(".", "")
            return " ".join(
                (name[0], name[1].upper() if len(name[1]) == 2 else name[1].title())
            )

        attrs_df["gauge_name"] = [fix_station_nm(n) for n in attrs_df["gauge_name"]]
        obj_cols = attrs_df.columns[attrs_df.dtypes == "object"]
        for c in obj_cols:
            attrs_df[c] = attrs_df[c].str.strip().astype(str)

        # transform categorical variables to numeric
        categorical_mappings = {}
        for column in attrs_df.columns:
            if attrs_df[column].dtype == "object":
                attrs_df[column] = attrs_df[column].astype("category")
                categorical_mappings[column] = dict(
                    enumerate(attrs_df[column].cat.categories)
                )
                attrs_df[column] = attrs_df[column].cat.codes

        # unify id to basin
        attrs_df.index.name = "basin"
        # We use xarray dataset to cache all data
        ds_from_df = attrs_df.to_xarray()
        units_dict = {
            "p_mean": "mm/day",
            "pet_mean": "mm/day",
            "aridity": "dimensionless",
            "p_seasonality": "dimensionless",
            "frac_snow": "dimensionless",
            "high_prec_freq": "days/yr",
            "high_prec_dur": "days",
            "high_prec_timing": "season",
            "low_prec_freq": "days/yr",
            "low_prec_dur": "days",
            "low_prec_timing": "season",
            "benchmark_catch": "Y/N",
            "surfacewater_abs": "mm/day",
            "groundwater_abs": "mm/day",
            "discharges": "mm/day",
            "abs_agriculture_perc": "percent",
            "abs_amenities_perc": "percent",
            "abs_energy_perc": "percent",
            "abs_environmental_perc": "percent",
            "abs_industry_perc": "percent",
            "abs_watersupply_perc": "percent",
            "num_reservoir": "dimensionless",
            "reservoir_cap": "ML",
            "reservoir_he": "percent",
            "reservoir_nav": "percent",
            "reservoir_drain": "percent",
            "reservoir_wr": "percent",
            "reservoir_fs": "percent",
            "reservoir_env": "percent",
            "reservoir_nousedata": "percent",
            "reservoir_year_first": "dimensionless",
            "reservoir_year_last": "dimensionless",
            "inter_high_perc": "percent",
            "inter_mod_perc": "percent",
            "inter_low_perc": "percent",
            "frac_high_perc": "percent",
            "frac_mod_perc": "percent",
            "frac_low_perc": "percent",
            "no_gw_perc": "percent",
            "low_nsig_perc": "percent",
            "nsig_low_perc": "percent",
            "q_mean": "mm/day",
            "runoff_ratio": "dimensionless",
            "stream_elas": "dimensionless",
            "slope_fdc": "dimensionless",
            "baseflow_index": "dimensionless",
            "baseflow_index_ceh": "dimensionless",
            "hfd_mean": "days since 1st October",
            "Q5": "mm/day",
            "Q95": "mm/day",
            "high_q_freq": "days/yr",
            "high_q_dur": "days",
            "low_q_freq": "days/yr",
            "low_q_dur": "days",
            "zero_q_freq": "percent",
            "station_type": "dimensionless",
            "flow_period_start": "dimensionless",
            "flow_period_end": "dimensionless",
            "flow_perc_complete": "percent",
            "bankfull_flow": "m3 sdimensionless1",
            "structurefull_flow": "m3 sdimensionless1",
            "q5_uncert_upper": "percent",
            "q5_uncert_lower": "percent",
            "q25_uncert_upper": "percent",
            "q25_uncert_lower": "percent",
            "q50_uncert_upper": "percent",
            "q50_uncert_lower": "percent",
            "q75_uncert_upper": "percent",
            "q75_uncert_lower": "percent",
            "q95_uncert_upper": "percent",
            "q95_uncert_lower": "percent",
            "q99_uncert_upper": "percent",
            "q99_uncert_lower": "percent",
            "quncert_meta": "dimensionless",
            "dwood_perc": "percent",
            "ewood_perc": "percent",
            "grass_perc": "percent",
            "shrub_perc": "percent",
            "crop_perc": "percent",
            "urban_perc": "percent",
            "inwater_perc": "percent",
            "bares_perc": "percent",
            "dom_land_cover": "dimensionless",
            "sand_perc": "percent",
            "sand_perc_missing": "percent",
            "silt_perc": "percent",
            "silt_perc_missing": "percent",
            "clay_perc": "percent",
            "clay_perc_missing": "percent",
            "organic_perc": "percent",
            "organic_perc_missing": "percent",
            "bulkdens": "g/cm^3",
            "bulkdens_missing": "percent",
            "bulkdens_5": "g/cm^3",
            "bulkdens_50": "g/cm^3",
            "bulkdens_95": "g/cm^3",
            "tawc": "mm",
            "tawc_missing": "percent",
            "tawc_5": "mm",
            "tawc_50": "mm",
            "tawc_95": "mm",
            "porosity_cosby": "dimensionless",
            "porosity_cosby_missing": "percent",
            "porosity_cosby_5": "dimensionless",
            "porosity_cosby_50": "dimensionless",
            "porosity_cosby_95": "dimensionless",
            "porosity_hypres": "dimensionless",
            "porosity_hypres_missing": "percent",
            "porosity_hypres_5": "dimensionless",
            "porosity_hypres_50": "dimensionless",
            "porosity_hypres_95": "dimensionless",
            "conductivity_cosby": "cm/h",
            "conductivity_cosby_missing": "percent",
            "conductivity_cosby_5": "cm/h",
            "conductivity_cosby_50": "cm/h",
            "conductivity_cosby_95": "cm/h",
            "conductivity_hypres": "cm/h",
            "conductivity_hypres_missing": "percent",
            "conductivity_hypres_5": "cm/h",
            "conductivity_hypres_50": "cm/h",
            "conductivity_hypres_95": "cm/h",
            "root_depth": "m",
            "root_depth_missing": "percent",
            "root_depth_5": "m",
            "root_depth_50": "m",
            "root_depth_95": "m",
            "soil_depth_pelletier": "m",
            "soil_depth_pelletier_missing": "percent",
            "soil_depth_pelletier_5": "m",
            "soil_depth_pelletier_50": "m",
            "soil_depth_pelletier_95": "m",
            "gauge_name": "dimensionless",
            "gauge_lat": "degree",
            "gauge_lon": "degree",
            "gauge_easting": "m",
            "gauge_northing": "m",
            "gauge_elev": "m.a.s.l",
            "area": "km^2",
            "dpsbar": "m/km",
            "elev_mean": "m.a.s.l",
            "elev_min": "m.a.s.l",
            "elev_10": "m.a.s.l",
            "elev_50": "m.a.s.l",
            "elev_90": "m.a.s.l",
            "elev_max": "m.a.s.l",
        }

        # Assign units to the variables in the Dataset
        for var_name in units_dict:
            if var_name in ds_from_df.data_vars:
                ds_from_df[var_name].attrs["units"] = units_dict[var_name]

        # Assign categorical mappings to the variables in the Dataset
        for column in ds_from_df.data_vars:
            if column in categorical_mappings:
                mapping_str = categorical_mappings[column]
                ds_from_df[column].attrs["category_mapping"] = str(mapping_str)

    def cache_forcing_xrdataset(self):
        """Save all basin-forcing data in a netcdf file in the cache directory.

        """
        cache_npy_file = CACHE_DIR.joinpath("camels_gb_forcing.npy")
        json_file = CACHE_DIR.joinpath("camels_gb_forcing.json")
        if (not os.path.isfile(cache_npy_file)) or (not os.path.isfile(json_file)):
            self.cache_forcing_np_json()
        forcing = np.load(cache_npy_file)
        with open(json_file, "r") as fp:
            forcing_dict = json.load(
                fp, object_pairs_hook=collections.OrderedDict
            )
        import pint_xarray

        basins = forcing_dict["basin"]
        times = pd.date_range(
            forcing_dict["time"][0], periods=len(forcing_dict["time"])
        )
        variables = forcing_dict["variable"]

        units = ["mm/day", "mm/day", "°C", "mm/day", "m^3/s", "mm/day", "g/kg", "W/m^2", "W/m^2", "m/s"]
        return xr.Dataset(
            data_vars={
                **{
                    variables[i]: (
                        ["basin", "time"],
                        forcing[:, :, i],
                        {"units": units[i]},
                    )
                    for i in range(len(variables))
                }
            },
            coords={
                "basin": basins,
                "time": times,
            },
            attrs={"forcing_type": "observation"},
        )

    def cache_streamflow_xrdataset(self):
        """Save all basins' streamflow data in a netcdf file in the cache directory

        """
        cache_npy_file = CACHE_DIR.joinpath("camels_gb_streamflow.npy")
        json_file = CACHE_DIR.joinpath("camels_gb_streamflow.json")
        if (not os.path.isfile(cache_npy_file)) or (not os.path.isfile(json_file)):
            self.cache_streamflow_np_json()
        streamflow = np.load(cache_npy_file)
        with open(json_file, "r") as fp:
            streamflow_dict = json.load(fp, object_pairs_hook=collections.OrderedDict)
        import pint_xarray

        basins = streamflow_dict["basin"]
        times = pd.date_range(
            streamflow_dict["time"][0], periods=len(streamflow_dict["time"])
        )
        return xr.Dataset(
            {
                "streamflow": (
                    ["basin", "time"],
                    streamflow[:, :, 0],
                    {"units": self.streamflow_unit},
                ),
                "ET": (
                    ["basin", "time"],
                    streamflow[:, :, 1],
                    {"units": "mm/day"},
                ),
            },
            coords={
                "basin": basins,
                "time": times,
            },
        )

    def cache_xrdataset(self):
        """
        Save all data in a netcdf file in the cache directory

        """

        warnings.warn("Check you units of all variables")
        ds_attr = self.cache_attributes_xrdataset()
        ds_attr.to_netcdf(CACHE_DIR.joinpath("camelsgb_attributes.nc"))
        ds_streamflow = self.cache_streamflow_xrdataset()
        ds_forcing = self.cache_forcing_xrdataset()
        ds = xr.merge([ds_streamflow, ds_forcing])
        ds.to_netcdf(CACHE_DIR.joinpath("camelsgb_timeseries.nc"))
