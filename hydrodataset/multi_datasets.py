"""
Read multiple camels series datasets
"""
import collections
from functools import reduce
import pandas as pd
import numpy as np
from hydroutils import hydro_time
from hydrodataset import (
    HydroDataset,
    DATASETS,
    REGIONS,
    Camels,
    CAMELS_NO_DATASET_ERROR_LOG,
)
from hydrodataset.lamah import Lamah

DATASETS_DICT = {
    "CAMELS": Camels,
    "LamaH": Lamah,
}


class MultiDatasets(HydroDataset):
    """A data source class for multiple datasets"""

    def __init__(
        self,
        data_path: list,
        download=False,
        datasets: list = None,
        regions: list = None,
    ):
        """

        Parameters
        ----------
        data_path:
            the paths of all necessary data
        download:
            if True, download the dataset
        datasets:
            a list with multiple datasets
        regions
            a list with multiple regions, each region corresponds to a dataset
        """
        if regions is None:
            regions = ["US"]
        if type(regions) != list:
            regions = [regions]
        if type(data_path) != list:
            data_path = [data_path]
        if not set(datasets).issubset(set(DATASETS)):
            raise NotImplementedError("We only support " + DATASETS + " now")
        if not set(regions).issubset(set(REGIONS)):
            raise NotImplementedError("We only support " + REGIONS + " now")
        if len(data_path) != len(regions):
            raise RuntimeError("Please choose directory for each region")
        for one_path in data_path:
            super().__init__(one_path)
        self.data_path = data_path
        self.datasets = datasets
        self.regions = regions
        self.data_source_description = self.set_data_source_describe()
        if download:
            self.download_data_source()
        self.sites = self.read_site_info()
        self.site_region_dict = self.read_site_id_dict()
        self.streamflow_dict = self.get_target_dict()
        self.forcing_dict = self.get_relevant_dict()
        self.attr_dict = self.get_constant_dict()
        self.str_attr_name = ["high_prec_timing", "low_prec_timing", "geol_1st_class"]

    def get_name(self):
        return "MULTI_DATASETS"

    def set_data_source_describe(self) -> collections.OrderedDict:
        describe = collections.OrderedDict({})
        for i in range(len(self.regions)):
            region = DATASETS_DICT[self.datasets[i]](
                self.data_path[i], False, region=self.regions[i]
            )
            describe = collections.OrderedDict(
                **describe, **{self.regions[i]: region.data_source_description}
            )
        return describe

    def download_data_source(self):
        for i in range(len(self.regions)):
            DATASETS_DICT[self.datasets[i]](
                self.data_path[i], True, region=self.regions[i]
            )

    def read_site_info(self) -> collections.OrderedDict:
        """
        Read the basic information of gages in multiple datasets

        Returns
        -------
        collections.OrderedDict
            basic info of gages for different regions
        """
        site_info = collections.OrderedDict({})
        for i in range(len(self.datasets)):
            region = DATASETS_DICT[self.datasets[i]](
                self.data_path[i], False, region=self.regions[i]
            )
            site_info = collections.OrderedDict(
                **site_info, **{self.regions[i]: region.read_site_info()}
            )
        return site_info

    def read_object_ids(self, object_params=None) -> np.array:
        region_id_lst = []
        for i in range(len(self.regions)):
            dataset = DATASETS_DICT[self.datasets[i]](
                self.data_path[i], False, region=self.regions[i]
            )
            region_id_lst = region_id_lst + dataset.read_object_ids().tolist()
        region_id_arr = np.array(region_id_lst)
        if np.unique(region_id_arr).size != region_id_arr.size:
            raise RuntimeError(
                "Same id for different sites, Error! Please check your chosen gages"
            )
        return region_id_arr

    def read_site_id_dict(self):
        site_id = collections.OrderedDict({})
        for i in range(len(self.regions)):
            region = DATASETS_DICT[self.datasets[i]](
                self.data_path[i], False, region=self.regions[i]
            )
            site_id = collections.OrderedDict(
                **site_id, **{self.regions[i]: region.read_object_ids()}
            )
        return site_id

    def read_target_cols(
        self, object_ids=None, t_range_list=None, target_cols=None, **kwargs
    ) -> np.array:
        """
        Read streamflow data

        We unify the unit of streamflow to ft3/s because first our data interface used in models are set for CAMELS-US.

        Parameters
        ----------
        object_ids
            sites
        t_range_list
            time range
        target_cols
            streamflow variable
        kwargs
            optional parameters

        Returns
        -------
        np.array
            streamflow data for given sites
        """
        nt = hydro_time.t_range_days(t_range_list).shape[0]
        flow = np.full((len(object_ids), nt, len(target_cols)), np.nan)
        for i in range(len(self.regions)):
            region_now = self.regions[i]
            dataset = DATASETS_DICT[self.datasets[i]](
                self.data_path[i], False, region=region_now
            )
            flow_dict_map = {target_cols[0]: self.streamflow_dict[region_now]}
            in_region_flow_tuple = [
                (k, value) for k, (key, value) in enumerate(flow_dict_map.items())
            ]
            sites_tuple = [
                (k, object_ids[k])
                for k in range(len(object_ids))
                if object_ids[k] in self.site_region_dict[region_now]
            ]
            sites_in_this_region = [a_tuple[1] for a_tuple in sites_tuple]
            in_region_flow = [index_attr[1] for index_attr in in_region_flow_tuple]
            flow_this_region = dataset.read_target_cols(
                sites_in_this_region, t_range_list, in_region_flow
            )
            sites_idx = [a_tuple[0] for a_tuple in sites_tuple]
            flow[sites_idx, :, :] = flow_this_region
        return flow

    def read_relevant_cols(
        self, object_ids=None, t_range_list=None, relevant_cols=None, **kwargs
    ) -> np.array:
        """
        Read data of common forcing variables between given regions

        Parameters
        ----------
        object_ids
            sites
        t_range_list
            time range
        relevant_cols
            common forcing variables for given regions
        kwargs
            optional parameters

        Returns
        -------
        np.array
            forcing data
        """
        nt = hydro_time.t_range_days(t_range_list).shape[0]
        forcings = np.full((len(object_ids), nt, len(relevant_cols)), np.nan)
        for i in range(len(self.regions)):
            region_now = self.regions[i]
            dataset = DATASETS_DICT[self.datasets[i]](
                self.data_path[i], False, region=region_now
            )
            region_forcing_all = dataset.get_relevant_cols()
            common_forcing_all = self.get_relevant_cols()
            j_index = [
                common_forcing_all.tolist().index(relevant_cols[ii])
                for ii in range(len(relevant_cols))
            ]
            forcing_dict_lst = [
                {relevant_cols[j]: self.forcing_dict[region_now][j_index[j]]}
                for j in range(len(relevant_cols))
            ]
            forcing_dict_map = reduce(
                lambda x, y: collections.OrderedDict(**x, **y), forcing_dict_lst
            )
            in_region_forcing_tuple = [
                (k, value)
                for k, (key, value) in enumerate(forcing_dict_map.items())
                if value in region_forcing_all
            ]
            not_in_region_forcing_tuple = [
                (k, value)
                for k, (key, value) in enumerate(forcing_dict_map.items())
                if value not in region_forcing_all
            ]
            sites_tuple = [
                (k, object_ids[k])
                for k in range(len(object_ids))
                if object_ids[k] in self.site_region_dict[region_now]
            ]
            sites_in_this_region = [a_tuple[1] for a_tuple in sites_tuple]
            in_region_forcing = [
                index_attr[1] for index_attr in in_region_forcing_tuple
            ]
            forcing1 = dataset.read_relevant_cols(
                sites_in_this_region, t_range_list, in_region_forcing
            )
            forcing2 = self.read_not_included_forcings(
                sites_in_this_region,
                t_range_list,
                [index_attr[1] for index_attr in not_in_region_forcing_tuple],
                dataset,
            )
            forcing1_idx = [
                index_forcing[0] for index_forcing in in_region_forcing_tuple
            ]
            forcing2_idx = [
                index_forcing[0] for index_forcing in not_in_region_forcing_tuple
            ]
            sites_idx = [a_tuple[0] for a_tuple in sites_tuple]
            forcing_this_region = np.empty(
                (len(sites_idx), nt, len(forcing1_idx) + len(forcing2_idx))
            )
            forcing_this_region[:, :, forcing1_idx] = forcing1
            forcing_this_region[:, :, forcing2_idx] = forcing2
            for j in range(len(relevant_cols)):
                # unit of srad in AUS is MJ/m2(/day) while others are W/m2
                if region_now == "AUS" and relevant_cols[j] == "srad":
                    forcing_this_region[:, :, j] = (
                        forcing_this_region[:, :, j] * 1e6 / (24 * 3600)
                    )
                # unit of vp in AUS is hPa while others are Pa
                if region_now == "AUS" and relevant_cols[j] == "vp":
                    forcing_this_region[:, :, j] = forcing_this_region[:, :, j] * 100
            forcings[sites_idx, :, :] = forcing_this_region
        return forcings

    @staticmethod
    def read_not_included_forcings(
        sites_id, t_range_list, forcing_cols, dataset
    ) -> np.array:
        """
        read forcings not included in the dataset of the region

        Parameters
        ----------
        sites_id

        t_range_list

        forcing_cols

        dataset
            data source


        Returns
        -------
        np.array
            forcings not included in original dataset
        """
        nt = hydro_time.t_range_days(t_range_list).shape[0]
        forcing_data = np.empty([len(sites_id), nt, len(forcing_cols)])
        for i in range(len(forcing_cols)):
            if forcing_cols[i] == "":
                forcing_data[:, :, i] = np.full((len(sites_id), nt), np.nan)
            elif forcing_cols[i] == "PET" and dataset.get_name() == "CAMELS_US":
                forcing_data[
                    :, :, i : i + 1
                ] = dataset.read_camels_us_model_output_data(
                    sites_id, t_range_list, ["PET"]
                )
            elif forcing_cols[i] == "PET_A" and dataset.get_name() == "LamaH_CE":
                forcing_data[
                    :, :, i : i + 1
                ] = dataset.read_lamah_hydro_model_time_series(
                    sites_id, t_range_list, ["PET_A"]
                )
        return forcing_data

    def read_constant_cols(
        self, object_ids=None, constant_cols: list = None, **kwargs
    ) -> np.array:
        """
        Read data of common attribute variables between given regions

        Parameters
        ----------
        object_ids
            sites
        constant_cols
            common attribute variables for given regions
        kwargs
            optional parameters

        Returns
        -------
        np.array
            attribute data
        """
        attrs = np.full((len(object_ids), len(constant_cols)), np.nan)
        str_attr_dict = {}
        for constant_col in constant_cols:
            if constant_col in self.str_attr_name:
                str_attr_dict = {
                    **str_attr_dict,
                    **{constant_col: np.empty(len(object_ids)).astype(str)},
                }
        for i in range(len(self.regions)):
            region_now = self.regions[i]
            dataset = DATASETS_DICT[self.datasets[i]](
                self.data_path[i], False, region=region_now
            )
            region_attr_all = dataset.get_constant_cols()
            common_attr_all = self.get_constant_cols()
            j_index = [
                common_attr_all.tolist().index(constant_col_)
                for constant_col_ in constant_cols
            ]
            constant_dict_lst = [
                {constant_cols[j]: self.attr_dict[region_now][j_index[j]]}
                for j in range(len(constant_cols))
            ]
            constant_dict_map = reduce(
                lambda x, y: collections.OrderedDict(**x, **y), constant_dict_lst
            )
            in_region_attr_tuple = [
                (k, value)
                for k, (key, value) in enumerate(constant_dict_map.items())
                if value in region_attr_all
            ]
            not_in_region_attr_tuple = [
                (k, value)
                for k, (key, value) in enumerate(constant_dict_map.items())
                if value not in region_attr_all
            ]
            sites_tuple = [
                (k, object_ids[k])
                for k in range(len(object_ids))
                if object_ids[k] in self.site_region_dict[region_now]
            ]
            sites_in_this_region = [a_tuple[1] for a_tuple in sites_tuple]
            in_region_attr = [index_attr[1] for index_attr in in_region_attr_tuple]
            if not set(in_region_attr).issubset(set(region_attr_all)):
                raise NotImplementedError(
                    "Wrong name for attributes, please check your input for attributes"
                )
            attrs1, var_dict, f_dict = dataset.read_constant_cols(
                sites_in_this_region, in_region_attr, is_return_dict=True
            )
            attrs2 = self.read_not_included_attrs(
                sites_in_this_region,
                [index_attr[1] for index_attr in not_in_region_attr_tuple],
                dataset,
            )
            attrs1_idx = [index_attr[0] for index_attr in in_region_attr_tuple]
            attrs2_idx = [index_attr[0] for index_attr in not_in_region_attr_tuple]
            sites_idx = [a_tuple[0] for a_tuple in sites_tuple]
            attrs_this_region = np.empty(
                (len(sites_idx), len(attrs1_idx) + len(attrs2_idx))
            )
            attrs_this_region[:, attrs1_idx] = attrs1
            attrs_this_region[:, attrs2_idx] = attrs2
            for j in range(len(constant_cols)):
                # restore to str
                if constant_cols[j] in self.str_attr_name:
                    # self.str_attr_name = ["high_prec_timing", "low_prec_timing", "geol_1st_class"]
                    restore_attr = np.array(
                        [
                            ""
                            if np.isnan(tmp)
                            else f_dict[list(constant_dict_lst[j].values())[0]][
                                int(tmp)
                            ]
                            for tmp in attrs_this_region[:, j]
                        ]
                    )
                    str_attr_dict[constant_cols[j]][sites_idx] = restore_attr
                # unit of slope_mean in AUS is % while others are m/km
                if region_now == "AUS" and constant_cols[j] == "slope_mean":
                    attrs_this_region[:, j] = attrs_this_region[:, j] * 10.0
                # unit of soil_depth in BR is cm while others are m
                if region_now == "BR" and constant_cols[j] == "soil_depth":
                    attrs_this_region[:, j] = attrs_this_region[:, j] / 100.0
                # unit of soil_conductivity in AUS is mm/h while others are cm/h
                if region_now == "AUS" and constant_cols[j] == "soil_conductivity":
                    attrs_this_region[:, j] = attrs_this_region[:, j] / 10.0
                # frac of gc_dom in CE need to be calculated
                if region_now == "CE" and constant_cols[j] == "geol_1st_class_frac":
                    geo_types_names_in_ce = [
                        "gc_ig_fra",
                        "gc_mt_fra",
                        "gc_pa_fra",
                        "gc_pb_fra",
                        "gc_pi_fra",
                        "gc_py_fra",
                        "gc_sc_fra",
                        "gc_sm_fra",
                        "gc_ss_fra",
                        "gc_su_fra",
                        "gc_va_fra",
                        "gc_vb_fra",
                        "gc_wb_fra",
                    ]
                    geo_fracs = dataset.read_constant_cols(
                        sites_in_this_region, geo_types_names_in_ce
                    )
                    attrs_this_region[:, j] = np.array(
                        [
                            geo_fracs[
                                k,
                                geo_types_names_in_ce.index(
                                    "gc_" + restore_attr[k] + "_fra"
                                ),
                            ]
                            for k in range(len(restore_attr))
                        ]
                    )
                # the type of geol_1st_class in GB are set "unknown", so all fractions are set to 100%
                if region_now == "GB" and constant_cols[j] == "geol_1st_class_frac":
                    attrs_this_region[:, j] = np.full(
                        attrs_this_region[:, j].shape, 1.0
                    )
            attrs[sites_idx, :] = attrs_this_region
        for j in range(len(constant_cols)):
            # trans str attr to number
            if constant_cols[j] in self.str_attr_name:
                if constant_cols[j] == "geol_1st_class":
                    geo_class_dict = {
                        "": "unknown",
                        "Acid plutonic rocks": "pa",
                        "Acid volcanic rocks": "va",
                        "Basic plutonic rocks": "pb",
                        "Basic volcanic rocks": "vb",
                        "CARBNATESED": "sc",
                        "Carbonate sedimentary rocks": "sc",
                        "IGNEOUS": "igneous",
                        "Ice and glaciers": "ig",
                        "Intermediate plutonic rocks": "pi",
                        "Intermediate volcanic rocks": "vi",
                        "METAMORPH": "mt",
                        "Metamorphics": "mt",
                        "Mixed sedimentary rocks": "sm",
                        "OTHERSED": "othersed",
                        "Pyroclastics": "py",
                        "SEDVOLC": "sm",
                        "SILICSED": "ss",
                        "Siliciclastic sedimentary rocks": "ss",
                        "UNCONSOLDTED": "su",
                        "Unconsolidated sediments": "su",
                        "Water bodies": "wb",
                        "acid_plutonic_rocks": "pa",
                        "acid_volcanic_rocks": "va",
                        "basic_volcanic_rocks": "vb",
                        "carbonate_sedimentary_rocks": "sc",
                        "intermediate_volcanic_rocks": "vi",
                        "metamorphics": "mt",
                        "mixed_sedimentary_rocks": "sm",
                        "mt": "mt",
                        "pa": "pa",
                        "pi": "pi",
                        "pyroclastics": "py",
                        "sc": "sc",
                        "siliciclastic_sedimentary_rocks": "ss",
                        "sm": "sm",
                        "ss": "ss",
                        "su": "su",
                        "unconsolidated_sediments": "su",
                        "vb": "vb",
                    }
                    str_attr_dict[constant_cols[j]] = np.array(
                        [geo_class_dict[tmp] for tmp in str_attr_dict[constant_cols[j]]]
                    )
                value, ref = pd.factorize(str_attr_dict[constant_cols[j]], sort=True)
                attrs[:, j] = value

        return attrs

    @staticmethod
    def read_not_included_attrs(sites_id, attr_cols, dataset) -> np.array:
        """
        read attributes not included in the dataset of the region

        Parameters
        ----------
        sites_id

        attr_cols

        dataset
            a dataset class instance


        Returns
        -------
        np.array
            attrs not included in original dataset
        """
        attr_data = np.empty([len(sites_id), len(attr_cols)])
        for i in range(len(attr_cols)):
            if attr_cols[i] == "":
                attr_data[:, i] = np.full(len(sites_id), np.nan)
            elif (
                attr_cols[i] == "dwood_perc+ewood_perc+shrub_perc"
                and dataset.get_name() == "CAMELS_GB"
            ):
                three_perc = dataset.read_constant_cols(
                    sites_id, ["dwood_perc", "ewood_perc", "shrub_perc"]
                )
                attr_data[:, i] = np.sum(three_perc, axis=1)
        return attr_data

    def get_constant_cols(self) -> np.array:
        """
        Only read common attribute variables between given regions

        Returns
        -------
        np.array
            common attribute variables for given regions
        """
        return np.array(
            [
                "p_mean",
                "pet_mean",
                "aridity",
                "p_seasonality",
                "frac_snow",
                "high_prec_freq",
                "high_prec_dur",
                "high_prec_timing",
                "low_prec_freq",
                "low_prec_dur",
                "low_prec_timing",  # climate 0-10
                "elev_mean",
                "slope_mean",
                "area",  # topography 11-13
                "forest_frac",  # land cover 14
                "soil_depth",
                "soil_conductivity",
                "sand_frac",
                "silt_frac",
                "clay_frac",  # soil 15-19
                "geol_1st_class",
                "geol_1st_class_frac",
            ]
        )  # geology; aus may have different classes for geology 20-21

    def get_constant_dict(self) -> collections.OrderedDict:
        """
        common attribute variables, get their real names in different datasets

        Returns
        -------
        collections.OrderedDict
            attribute variables for all regions
        """
        attr_dict = collections.OrderedDict({})
        for i in range(len(self.regions)):
            if self.regions[i] == "AUS":
                attr_dict = collections.OrderedDict(
                    **attr_dict,
                    **{
                        self.regions[i]: [
                            "p_mean",
                            "pet_mean",
                            "aridity",
                            "p_seasonality",
                            "frac_snow",
                            "high_prec_freq",
                            "high_prec_dur",
                            "high_prec_timing",
                            "low_prec_freq",
                            "low_prec_dur",
                            "low_prec_timing",
                            "elev_mean",
                            "mean_slope_pct",
                            "catchment_area",
                            "prop_forested",
                            "solum_thickness",
                            "ksat",
                            "sanda",
                            "",
                            "claya",
                            "geol_prim",
                            "geol_prim_prop",
                        ]
                    }
                )
            elif self.regions[i] == "BR":
                attr_dict = collections.OrderedDict(
                    **attr_dict,
                    **{
                        self.regions[i]: [
                            "p_mean",
                            "pet_mean",
                            "aridity",
                            "p_seasonality",
                            "frac_snow",
                            "high_prec_freq",
                            "high_prec_dur",
                            "high_prec_timing",
                            "low_prec_freq",
                            "low_prec_dur",
                            "low_prec_timing",
                            "elev_mean",
                            "slope_mean",
                            "area",
                            "forest_perc",
                            "bedrock_depth",
                            "",
                            "sand_perc",
                            "silt_perc",
                            "clay_perc",
                            "geol_class_1st",
                            "geol_class_1st_perc",
                        ]
                    }
                )
            elif self.regions[i] == "CL":
                attr_dict = collections.OrderedDict(
                    **attr_dict,
                    **{
                        self.regions[i]: [
                            "p_mean_cr2met",
                            "pet_mean",
                            "aridity_cr2met",
                            "p_seasonality_cr2met",
                            "frac_snow_cr2met",
                            "high_prec_freq_cr2met",
                            "high_prec_dur_cr2met",
                            "high_prec_timing_cr2met",
                            "low_prec_freq_cr2met",
                            "low_prec_dur_cr2met",
                            "low_prec_timing_cr2met",
                            "elev_mean",
                            "slope_mean",
                            "area",
                            "forest_frac",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "geol_class_1st",
                            "geol_class_1st_frac",
                        ]
                    }
                )
            elif self.regions[i] == "GB":
                attr_dict = collections.OrderedDict(
                    **attr_dict,
                    **{
                        self.regions[i]: [
                            "p_mean",
                            "pet_mean",
                            "aridity",
                            "p_seasonality",
                            "frac_snow",
                            "high_prec_freq",
                            "high_prec_dur",
                            "high_prec_timing",
                            "low_prec_freq",
                            "low_prec_dur",
                            "low_prec_timing",
                            "elev_mean",
                            "dpsbar",
                            "area",
                            "dwood_perc+ewood_perc+shrub_perc",
                            "soil_depth_pelletier",
                            "conductivity_cosby",
                            "sand_perc",
                            "silt_perc",
                            "clay_perc",
                            "",
                            "",
                        ]
                    }
                )
            elif self.regions[i] == "US":
                attr_dict = collections.OrderedDict(
                    **attr_dict,
                    **{
                        self.regions[i]: [
                            "p_mean",
                            "pet_mean",
                            "aridity",
                            "p_seasonality",
                            "frac_snow",
                            "high_prec_freq",
                            "high_prec_dur",
                            "high_prec_timing",
                            "low_prec_freq",
                            "low_prec_dur",
                            "low_prec_timing",
                            "elev_mean",
                            "slope_mean",
                            "area_gages2",
                            "frac_forest",
                            "soil_depth_pelletier",
                            "soil_conductivity",
                            "sand_frac",
                            "silt_frac",
                            "clay_frac",
                            "geol_1st_class",
                            "glim_1st_class_frac",
                        ]
                    }
                )
            elif self.regions[i] == "YR":
                raise NotImplementedError(
                    "Yellow river only provided normalized streamflow data, so please don't use it now."
                )
            elif self.regions[i] == "CA":
                raise NotImplementedError(
                    "Only a few attributes are provided in CA region, so please don't use it now."
                )
            elif self.regions[i] == "CE":
                attr_dict = collections.OrderedDict(
                    **attr_dict,
                    **{
                        self.regions[i]: [
                            "p_mean",
                            "et0_mean",
                            "arid_1",
                            "p_season",
                            "frac_snow",
                            "hi_prec_fr",
                            "hi_prec_du",
                            "hi_prec_ti",
                            "lo_prec_fr",
                            "lo_prec_du",
                            "lo_prec_ti",
                            "elev_mean",
                            "slope_mean",
                            "area_calc",
                            "forest_fra",
                            "bedrk_dep",
                            "soil_condu",
                            "sand_fra",
                            "silt_fra",
                            "clay_fra",
                            "gc_dom",
                            "",
                        ]
                    }
                )
            else:
                raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)
        return attr_dict

    def get_relevant_cols(self):
        """
        Only read common forcing variables between given regions

        Returns
        -------
        np.array
            common forcing variables for given regions
        """
        return np.array(
            ["aet", "pet", "prcp", "srad", "swe", "tmax", "tmean", "tmin", "vp"]
        )

    def get_relevant_dict(self) -> collections.OrderedDict:
        """
        common forcing variables, get their real names in different datasets

        Returns
        -------
        collections.OrderedDict
            forcing variables for all regions
        """
        forcing_dict = collections.OrderedDict({})
        for i in range(len(self.regions)):
            if self.regions[i] == "AUS":
                # we use SILO data here
                forcing_dict = collections.OrderedDict(
                    **forcing_dict,
                    **{
                        self.regions[i]: [
                            "et_morton_actual_SILO",
                            "et_morton_point_SILO",
                            "precipitation_SILO",
                            "radiation_SILO",
                            "",
                            "tmax_SILO",
                            "",
                            "tmin_SILO",
                            "vp_SILO",
                        ]
                    }
                )
            elif self.regions[i] == "BR":
                forcing_dict = collections.OrderedDict(
                    **forcing_dict,
                    **{
                        self.regions[i]: [
                            "evapotransp_gleam",
                            "potential_evapotransp_gleam",
                            "precipitation_chirps",
                            "",
                            "",
                            "temperature_max_cpc",
                            "temperature_mean_cpc",
                            "temperature_min_cpc",
                            "",
                        ]
                    }
                )
            elif self.regions[i] == "CL":
                forcing_dict = collections.OrderedDict(
                    **forcing_dict,
                    **{
                        self.regions[i]: [
                            "",
                            "pet_hargreaves",
                            "precip_cr2met",
                            "",
                            "swe",
                            "tmax_cr2met",
                            "tmean_cr2met",
                            "tmin_cr2met",
                            "",
                        ]
                    }
                )
            elif self.regions[i] == "GB":
                forcing_dict = collections.OrderedDict(
                    **forcing_dict,
                    **{
                        self.regions[i]: [
                            "",
                            "pet",
                            "precipitation",
                            "shortwave_rad",
                            "",
                            "",
                            "temperature",
                            "",
                            "",
                        ]
                    }
                )
            elif self.regions[i] == "US":
                forcing_dict = collections.OrderedDict(
                    **forcing_dict,
                    **{
                        self.regions[i]: [
                            "",
                            "PET",
                            "prcp",
                            "srad",
                            "swe",
                            "tmax",
                            "",
                            "tmin",
                            "vp",
                        ]
                    }
                )
            elif self.regions[i] == "YR":
                raise NotImplementedError(
                    "Yellow river only provided normalized streamflow data, so please don't use it now."
                )
            elif self.regions[i] == "CA":
                forcing_dict = collections.OrderedDict(
                    **forcing_dict,
                    **{
                        self.regions[i]: [
                            "",
                            "",
                            "prcp",
                            "",
                            "",
                            "tmax",
                            "",
                            "tmin",
                            "",
                        ]
                    }
                )
            elif self.regions[i] == "CE":
                forcing_dict = collections.OrderedDict(
                    **forcing_dict,
                    **{
                        self.regions[i]: [
                            "total_et",
                            "PET_A",
                            "prec",
                            "surf_net_solar_rad_mean",
                            "swe",
                            "2m_temp_max",
                            "2m_temp_mean",
                            "2m_temp_min",
                            "",
                        ]
                    }
                )
            else:
                raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)
        return forcing_dict

    def get_target_cols(self) -> np.array:
        """
        Only read common target variables -- streamflow

        Returns
        -------
        np.array
            common target variables for given regions
        """
        # unify all "streamflow" names to "streamflow"
        return np.array(["streamflow"])

    def get_target_dict(self) -> collections.OrderedDict:
        """
        common target variable -- streamflow, get its real name in different datasets

        Returns
        -------
        collections.OrderedDict
            streamflow variables for all regions
        """
        streamflow_dict = collections.OrderedDict({})
        for i in range(len(self.regions)):
            if self.regions[i] == "US":
                # its unit is ft3/s
                streamflow_dict = collections.OrderedDict(
                    **streamflow_dict, **{self.regions[i]: "usgsFlow"}
                )
            elif self.regions[i] == "AUS":
                # MLd means "1 Megaliters Per Day"; 1 MLd = 0.011574074074074 cubic-meters-per-second
                streamflow_dict = collections.OrderedDict(
                    **streamflow_dict, **{self.regions[i]: "streamflow_MLd"}
                )
            elif self.regions[i] in ["BR", "CL"]:
                streamflow_dict = collections.OrderedDict(
                    **streamflow_dict, **{self.regions[i]: "streamflow_m3s"}
                )
            elif self.regions[i] == "GB":
                streamflow_dict = collections.OrderedDict(
                    **streamflow_dict, **{self.regions[i]: "discharge_vol"}
                )
            elif self.regions[i] == "YR":
                raise NotImplementedError(
                    "Yellow river only provided normalized streamflow data, so please don't use it now."
                )
            elif self.regions[i] == "CA":
                # mm/day, remember to trans to m3/s
                streamflow_dict = collections.OrderedDict(
                    **streamflow_dict, **{self.regions[i]: "discharge"}
                )
            elif self.regions[i] == "CE":
                streamflow_dict = collections.OrderedDict(
                    **streamflow_dict, **{self.regions[i]: "qobs"}
                )
            else:
                raise NotImplementedError(CAMELS_NO_DATASET_ERROR_LOG)
        return streamflow_dict

    def get_other_cols(self) -> dict:
        pass

    def read_other_cols(self, object_ids=None, other_cols=None, **kwargs):
        pass

    def read_area(self, object_ids) -> np.array:
        return self.read_constant_cols(object_ids, ["area"], is_return_dict=False)

    def read_mean_prep(self, object_ids) -> np.array:
        return self.read_constant_cols(object_ids, ["p_mean"], is_return_dict=False)
