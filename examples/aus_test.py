import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from hydrodataset import Camels

directory = 'basin_flow'
if not os.path.exists(directory):
    os.makedirs(directory)

camels_aus_path = "/ftproot/camels/camels_aus/"
camels_aus_v2_path="/home/estelle/data/waterism/datasets-origin/camels/camels_aus_v2/"

aus_region = "AUS"
aus_v2_region = "AUS_v2"
# ------------------------------ AUS --------------------------------
camels_aus = Camels(camels_aus_path, download=False, region=aus_region)
camels_aus_v2=Camels(camels_aus_v2_path, download=False, region=aus_v2_region)
gage_ids = camels_aus.read_object_ids()

# Return -> np.array: forcing data
hydro_info = camels_aus.read_relevant_cols(
    gage_ids[:],
    ["2015-01-01", "2022-02-15"],
    ["et_morton_point_SILO", "precipitation_SILO", "et_morton_actual_SILO"]
)

gages_to_nan = ['403213A', '224213A', '224214A', '227225A']

# 1 Megaliters Per Day = 0.011574074074074 Cubic Meters Per Second
# ML_to_m3_per_s = 0.011574074074074

t_info = pd.date_range(start="2015-01-01", end="2022-02-14", freq='D')
formatted_time = t_info.strftime('%Y-%m-%d %H:%M:%S')

for i, gage_id in enumerate(gage_ids):
    hydro_data = hydro_info[i]

    if gage_id in gages_to_nan:
        streamflow_data_m3_per_s = np.nan * np.ones_like(hydro_data[:, 0])
    else:
        # Return -> np.array: streamflow data, 3-dim [station, time, streamflow(ML/d)]
        streamflow_info = camels_aus_v2.read_target_cols(
            gage_ids[i:i+1],
            ["2015-01-01", "2022-02-15"],
            target_cols=["streamflow_MLd"],
        )
        streamflow_data_m3_per_s = (streamflow_info[0,:,0]/35.314666721489)

    pet = hydro_data[:, 0]
    prcp = hydro_data[:, 1]
    flow = streamflow_data_m3_per_s
    et = hydro_data[:, 2]
    node1_flow = np.nan * np.ones_like(flow)  # NA for node1_flow
    merged_row = np.column_stack([formatted_time, pet, prcp, flow, et, node1_flow])
    # tiem pet(mm/day) prcp(mm/day) flow(m^3/s) et(mm/day) node1_flow(m^3/s)
    columns = ["time", "pet(mm/day)", "prcp(mm/day)", "flow(m^3/s)", "et(mm/day)", "node1_flow(m^3/s)"]
    df = pd.DataFrame(merged_row, columns=columns)
    filename = f'basin_{gage_id}.csv'
    file_path = os.path.join(directory, filename)
    df.to_csv(file_path, index=False)

