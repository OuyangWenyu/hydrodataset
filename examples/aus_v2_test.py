import os
import numpy as np

from hydrodataset import Camels

camels_aus_v2_path="/home/estelle/data/waterism/datasets-origin/camels/camels_aus_v2/"

aus_v2_region = "AUS_v2"

# ---------------------------- AUS-V2 -------------------------------
camels_aus_v2=Camels(camels_aus_v2_path, download=False, region=aus_v2_region)

gage_ids = camels_aus_v2.read_object_ids()

p_mean_info=camels_aus_v2.read_mean_prcp(
    gage_ids[:5],unit="mm/h"
)
print(p_mean_info)

attrs = camels_aus_v2.read_constant_cols(
    gage_ids[:5], var_lst=["catchment_area", "geol_sec", "metamorph"]
)
print(attrs)
forcings = camels_aus_v2.read_relevant_cols(
    gage_ids[:5],
    ["1990-01-01", "2010-01-01"],
    var_lst=["precipitation_AGCD", "et_morton_actual_SILO", "tmin_SILO"],
)
print(forcings.shape)
flows = camels_aus_v2.read_target_cols(
    gage_ids[:5],
    ["2015-01-01", "2022-01-01"],
    target_cols=["streamflow_MLd", "streamflow_mmd"],
)
print(flows)
streamflow_types = camels_aus_v2.get_target_cols()
print(streamflow_types)
focing_types = camels_aus_v2.get_relevant_cols()
print(focing_types)
attr_types = camels_aus_v2.get_constant_cols()
print(attr_types)