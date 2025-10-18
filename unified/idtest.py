from hydrodataset import SETTING, CACHE_DIR
from hydrodataset.camelsh import Camelsh  # 从 camelsh.py 导入
from hydrodataset.camels_aus_aqua import CamelsAus  # 从 camels_aus_aqua.py 导入
from hydrodataset.camels_cl_aqua import CamelsCl  # 从 camels_cl_aqua.py 导入
from hydrodataset.camels_dk_aqua import CamelsDK  # 从 camels_dk_aqua.py 导入
from hydrodataset.camels_col_aqua import CamelsCol  # 从 camels_col_aqua.py 导入
from hydrodataset.camels_se_aqua import CamelsSe  # 从 camels_se_aqua.py 导入
from hydrodataset.camels_sk_aqua import CamelsSk  # 从 camels_sk_aqua.py 导入
from hydrodataset.camels_gb_aqua import CamelsGb  # 从 camels_gb_aqua.py 导入
from hydrodataset.camels_fi_aqua import CamelsFi  # 从 camels_fi_aqua.py 导入
from hydrodataset.camels_lux_aqua import CamelsLux  # 从 camels_lux_aqua.py 导入
from hydrodataset.camels_nz_aqua import CamelsNz  # 从 camels_nz_aqua.py 导入
from hydrodataset.camels_de_aqua import CamelsDe  # 从 camels_de_aqua.py 导入
from hydrodataset.camels_fr_aqua import CamelsFr  # 从 camels_fr_aqua.py 导入
from hydrodataset.camels_ch_aqua import CamelsCh  # 从 camels_ch_aqua.py 导入
import pandas as pd
import xarray as xr


data_path = SETTING["local_data_path"]["root"]

ds_camelsh = Camelsh(data_path)
ds_camels_aus = CamelsAus(data_path)
ds_camels_cl = CamelsCl(data_path)
ds_camels_dk = CamelsDK(data_path)
ds_camels_col = CamelsCol(data_path)
ds_camels_se = CamelsSe(data_path)
ds_camels_sk = CamelsSk(data_path)
ds_camels_gb = CamelsGb(data_path)
ds_camels_fi = CamelsFi(data_path)
ds_camels_lux = CamelsLux(data_path)
ds_camels_nz = CamelsNz(data_path)
ds_camels_de = CamelsDe(data_path)
ds_camels_fr = CamelsFr(data_path)
ds_camels_ch = CamelsCh(data_path)

camelsh_ids = ds_camelsh.read_object_ids()
camels_aus_ids = ds_camels_aus.read_object_ids()
camels_cl_ids = ds_camels_cl.read_object_ids()
camels_dk_ids = ds_camels_dk.read_object_ids()
camels_col_ids = ds_camels_col.read_object_ids()
camels_se_ids = ds_camels_se.read_object_ids()
camels_sk_ids = ds_camels_sk.read_object_ids()
camels_gb_ids = ds_camels_gb.read_object_ids()
camels_fi_ids = ds_camels_fi.read_object_ids()
camels_lux_ids = ds_camels_lux.read_object_ids()
camels_nz_ids = ds_camels_nz.read_object_ids()
camels_de_ids = ds_camels_de.read_object_ids()
camels_fr_ids = ds_camels_fr.read_object_ids()
camels_ch_ids = ds_camels_ch.read_object_ids()

# 创建数据集ID字典，便于比较
datasets = {
    'CAMELS-H': camelsh_ids,
    'CAMELS-AUS': camels_aus_ids,
    'CAMELS-CL': camels_cl_ids,
    'CAMELS-DK': camels_dk_ids,
    'CAMELS-COL': camels_col_ids,
    'CAMELS-SE': camels_se_ids,
    'CAMELS-SK': camels_sk_ids,
    'CAMELS-GB': camels_gb_ids,
    'CAMELS-FI': camels_fi_ids,
    'CAMELS-LUX': camels_lux_ids,
    'CAMELS-NZ': camels_nz_ids,
    'CAMELS-DE': camels_de_ids,
    'CAMELS-FR': camels_fr_ids,
    'CAMELS-CH': camels_ch_ids,
}

print("=== 数据集ID统计信息 ===")
for name, ids in datasets.items():
    print(f"{name}: {len(ids)} 个站点")

print("\n=== ID重复情况分析 ===")


# 1. 检查所有数据集之间的ID重复情况
def find_duplicate_ids(datasets):
    """查找不同数据集之间的重复ID"""
    duplicate_pairs = []
    dataset_names = list(datasets.keys())

    for i in range(len(dataset_names)):
        for j in range(i + 1, len(dataset_names)):
            name1, name2 = dataset_names[i], dataset_names[j]
            ids1, ids2 = datasets[name1], datasets[name2]

            # 转换为集合进行交集运算
            set1, set2 = set(ids1), set(ids2)
            common_ids = set1.intersection(set2)

            if common_ids:
                duplicate_pairs.append(
                    {
                        'dataset1': name1,
                        'dataset2': name2,
                        'common_ids': list(common_ids),
                        'count': len(common_ids),
                    }
                )

    return duplicate_pairs


duplicate_pairs = find_duplicate_ids(datasets)

if duplicate_pairs:
    print("发现以下数据集之间存在重复ID:")
    for pair in duplicate_pairs:
        print(f"  {pair['dataset1']} 与 {pair['dataset2']}: {pair['count']} 个重复ID")
        if pair['count'] <= 10:  # 如果重复ID数量不多，显示具体ID
            print(f"    重复的ID: {pair['common_ids']}")
        else:
            print(f"    前10个重复ID: {pair['common_ids'][:10]}")
        print()
else:
    print("未发现任何数据集之间存在重复ID")

# 2. 统计每个ID在多少个数据集中出现
print("=== ID出现频率统计 ===")
all_ids = set()
for ids in datasets.values():
    all_ids.update(ids)

id_frequency = {}
for dataset_name, ids in datasets.items():
    for id_val in ids:
        if id_val not in id_frequency:
            id_frequency[id_val] = []
        id_frequency[id_val].append(dataset_name)

# 统计ID出现次数
id_counts = {id_val: len(datasets) for id_val, datasets in id_frequency.items()}
frequency_distribution = {}
for count in id_counts.values():
    frequency_distribution[count] = frequency_distribution.get(count, 0) + 1

print("ID出现频率分布:")
for count in sorted(frequency_distribution.keys()):
    print(f"  出现在 {count} 个数据集中的ID数量: {frequency_distribution[count]}")

# 3. 找出出现在多个数据集中的ID
multi_dataset_ids = {
    id_val: datasets for id_val, datasets in id_frequency.items() if len(datasets) > 1
}
if multi_dataset_ids:
    print(f"\n出现在多个数据集中的ID (共{len(multi_dataset_ids)}个):")
    for id_val, dataset_list in list(multi_dataset_ids.items()):
        print(f"  ID {id_val}: {', '.join(dataset_list)}")
