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


class UnifiedCamelsDataset:
    # 类属性：缓存站点列表（初始化时加载一次）
    _camelsh_stations = None
    _camels_aus_stations = None
    _camels_cl_stations = None
    _camels_dk_stations = None
    _camels_col_stations = None
    _camels_se_stations = None
    _camels_sk_stations = None
    _camels_gb_stations = None
    _camels_fi_stations = None
    _camels_lux_stations = None
    _camels_nz_stations = None
    _camels_de_stations = None
    _camels_fr_stations = None
    _camels_ch_stations = None

    def __init__(self, data_path=None):
        """
        初始化统一数据集类（不指定类型，用于自动检测）。

        参数:
        - data_path: 数据根路径（默认从 SETTING 获取）
        """
        if data_path is None:
            self.data_path = SETTING["local_data_path"]["root"]
        else:
            self.data_path = data_path

        # 加载站点列表（只加载一次）
        if UnifiedCamelsDataset._camelsh_stations is None:
            camelsh_ds = Camelsh(self.data_path)
            UnifiedCamelsDataset._camelsh_stations = set(
                camelsh_ds.read_object_ids()
            )  # 使用 set 加速查找

        if UnifiedCamelsDataset._camels_aus_stations is None:
            camels_aus_ds = CamelsAus(self.data_path)
            UnifiedCamelsDataset._camels_aus_stations = set(
                camels_aus_ds.read_object_ids()
            )

        if UnifiedCamelsDataset._camels_cl_stations is None:
            camels_cl_ds = CamelsCl(self.data_path)
            UnifiedCamelsDataset._camels_cl_stations = set(
                camels_cl_ds.read_object_ids()
            )

        if UnifiedCamelsDataset._camels_dk_stations is None:
            camels_dk_ds = CamelsDK(self.data_path)
            UnifiedCamelsDataset._camels_dk_stations = set(
                camels_dk_ds.read_object_ids()
            )

        if UnifiedCamelsDataset._camels_col_stations is None:
            camels_col_ds = CamelsCol(self.data_path)
            UnifiedCamelsDataset._camels_col_stations = set(
                camels_col_ds.read_object_ids()
            )

        if UnifiedCamelsDataset._camels_se_stations is None:
            camels_se_ds = CamelsSe(self.data_path)
            UnifiedCamelsDataset._camels_se_stations = set(
                camels_se_ds.read_object_ids()
            )

        if UnifiedCamelsDataset._camels_sk_stations is None:
            camels_sk_ds = CamelsSk(self.data_path)
            UnifiedCamelsDataset._camels_sk_stations = set(
                camels_sk_ds.read_object_ids()
            )

        if UnifiedCamelsDataset._camels_gb_stations is None:
            camels_gb_ds = CamelsGb(self.data_path)
            UnifiedCamelsDataset._camels_gb_stations = set(
                camels_gb_ds.read_object_ids()
            )

        if UnifiedCamelsDataset._camels_fi_stations is None:
            camels_fi_ds = CamelsFi(self.data_path)
            UnifiedCamelsDataset._camels_fi_stations = set(
                camels_fi_ds.read_object_ids()
            )

        if UnifiedCamelsDataset._camels_lux_stations is None:
            camels_lux_ds = CamelsLux(self.data_path)
            UnifiedCamelsDataset._camels_lux_stations = set(
                camels_lux_ds.read_object_ids()
            )

        if UnifiedCamelsDataset._camels_nz_stations is None:
            camels_nz_ds = CamelsNz(self.data_path)
            UnifiedCamelsDataset._camels_nz_stations = set(
                camels_nz_ds.read_object_ids()
            )

        if UnifiedCamelsDataset._camels_de_stations is None:
            camels_de_ds = CamelsDe(self.data_path)
            UnifiedCamelsDataset._camels_de_stations = set(
                camels_de_ds.read_object_ids()
            )

        if UnifiedCamelsDataset._camels_fr_stations is None:
            camels_fr_ds = CamelsFr(self.data_path)
            UnifiedCamelsDataset._camels_fr_stations = set(
                camels_fr_ds.read_object_ids()
            )

        if UnifiedCamelsDataset._camels_ch_stations is None:
            camels_ch_ds = CamelsCh(self.data_path)
            UnifiedCamelsDataset._camels_ch_stations = set(
                camels_ch_ds.read_object_ids()
            )

    def get_station_info(
        self,
        station_id_lst: list[str],
        t_range: list = None,
        static_lst: list = None,
        dynamic_lst: list = None,
    ) -> dict[str, dict]:
        """
        根据站点 ID 列表 获取信息（支持多个 ID，包括多数据集ID）。

        参数:
        - station_id_lst: 站点 ID 列表 (list[str]) 或单个 str
        - t_range: 时间范围 (list)，可选
        - static_lst: 静态属性列表 (list)，可选
        - dynamic_lst: 动态变量列表 (list)，可选

        返回:
        - dict: {station_id: {'datasets': {dataset_name: {'static_attrs': DataFrame, 'timeseries': DataFrame or None}}}}
        """
        if isinstance(station_id_lst, str):  # 支持单个 ID（转换为列表）
            station_id_lst = [station_id_lst]

        results = {}  # 结果字典: {id: info}

        for station_id in station_id_lst:
            try:
                # 检测ID存在于哪些数据集中
                datasets_for_id = self._find_datasets_for_id(station_id)

                if not datasets_for_id:
                    print(f"警告: 跳过无效站点 ID '{station_id}'（不属于任何数据集）。")
                    continue

                info = {'datasets': {}}

                # 为每个包含该ID的数据集获取信息
                for dataset_name, dataset_class in datasets_for_id.items():
                    try:
                        dataset_instance = dataset_class(self.data_path)

                        dataset_info = {}

                        # 获取静态属性
                        attrs_ds = dataset_instance.read_attr_xrdataset(
                            gage_id_lst=[station_id], var_lst=static_lst
                        )
                        dataset_info['static_attrs'] = (
                            attrs_ds.to_dataframe().reset_index()
                        )

                        # 获取时间序列
                        ts_ds = dataset_instance.read_ts_xrdataset(
                            gage_id_lst=[station_id],
                            t_range=t_range,
                            var_lst=dynamic_lst,
                        )
                        dataset_info['timeseries'] = ts_ds.to_dataframe().reset_index()

                        info['datasets'][dataset_name] = dataset_info

                    except Exception as e:
                        print(
                            f"错误: 在数据集 {dataset_name} 中处理站点 '{station_id}' 失败 - {e}"
                        )
                        info['datasets'][dataset_name] = {
                            'static_attrs': None,
                            'timeseries': None,
                        }

                results[station_id] = info
            except Exception as e:
                print(f"错误: 处理站点 '{station_id}' 失败 - {e}")

        return results

    def _find_datasets_for_id(self, station_id: str) -> dict[str, type]:
        """
        查找站点ID存在于哪些数据集中。

        参数:
        - station_id: 站点ID

        返回:
        - dict[str, type]: {dataset_name: dataset_class}
        """
        datasets = {}

        if station_id in UnifiedCamelsDataset._camelsh_stations:
            datasets['CAMELS-H'] = Camelsh
        if station_id in UnifiedCamelsDataset._camels_aus_stations:
            datasets['CAMELS-AUS'] = CamelsAus
        if station_id in UnifiedCamelsDataset._camels_cl_stations:
            datasets['CAMELS-CL'] = CamelsCl
        if station_id in UnifiedCamelsDataset._camels_dk_stations:
            datasets['CAMELS-DK'] = CamelsDK
        if station_id in UnifiedCamelsDataset._camels_col_stations:
            datasets['CAMELS-COL'] = CamelsCol
        if station_id in UnifiedCamelsDataset._camels_se_stations:
            datasets['CAMELS-SE'] = CamelsSe
        if station_id in UnifiedCamelsDataset._camels_sk_stations:
            datasets['CAMELS-SK'] = CamelsSk
        if station_id in UnifiedCamelsDataset._camels_gb_stations:
            datasets['CAMELS-GB'] = CamelsGb
        if station_id in UnifiedCamelsDataset._camels_fi_stations:
            datasets['CAMELS-FI'] = CamelsFi
        if station_id in UnifiedCamelsDataset._camels_lux_stations:
            datasets['CAMELS-LUX'] = CamelsLux
        if station_id in UnifiedCamelsDataset._camels_nz_stations:
            datasets['CAMELS-NZ'] = CamelsNz
        if station_id in UnifiedCamelsDataset._camels_de_stations:
            datasets['CAMELS-DE'] = CamelsDe
        if station_id in UnifiedCamelsDataset._camels_fr_stations:
            datasets['CAMELS-FR'] = CamelsFr
        if station_id in UnifiedCamelsDataset._camels_ch_stations:
            datasets['CAMELS-CH'] = CamelsCh

        return datasets

    def detect_multi_dataset_ids(
        self, station_id_lst: list[str] = None
    ) -> dict[str, list[str]]:
        """
        检测哪些ID存在于多个数据集中。

        参数:
        - station_id_lst: 要检查的站点ID列表，如果为None则检查所有ID

        返回:
        - dict[str, list[str]]: {station_id: [dataset_names]}
        """
        if station_id_lst is None:
            # 获取所有ID
            all_ids = set()
            for stations in [
                UnifiedCamelsDataset._camelsh_stations,
                UnifiedCamelsDataset._camels_aus_stations,
                UnifiedCamelsDataset._camels_cl_stations,
                UnifiedCamelsDataset._camels_dk_stations,
                UnifiedCamelsDataset._camels_col_stations,
                UnifiedCamelsDataset._camels_se_stations,
                UnifiedCamelsDataset._camels_sk_stations,
                UnifiedCamelsDataset._camels_gb_stations,
                UnifiedCamelsDataset._camels_fi_stations,
                UnifiedCamelsDataset._camels_lux_stations,
                UnifiedCamelsDataset._camels_nz_stations,
                UnifiedCamelsDataset._camels_de_stations,
                UnifiedCamelsDataset._camels_fr_stations,
                UnifiedCamelsDataset._camels_ch_stations,
            ]:
                all_ids.update(stations)
            station_id_lst = list(all_ids)

        if isinstance(station_id_lst, str):
            station_id_lst = [station_id_lst]

        multi_dataset_ids = {}
        for station_id in station_id_lst:
            datasets_for_id = self._find_datasets_for_id(station_id)
            if len(datasets_for_id) > 1:
                multi_dataset_ids[station_id] = list(datasets_for_id.keys())

        return multi_dataset_ids

    def get_dataset_count_for_id(self, station_id: str) -> int:
        """
        获取站点ID存在于多少个数据集中。

        参数:
        - station_id: 站点ID

        返回:
        - int: 数据集数量
        """
        datasets_for_id = self._find_datasets_for_id(station_id)
        return len(datasets_for_id)

    def print_multi_dataset_info(self, station_id_lst: list[str] = None):
        """
        打印多数据集ID的信息。

        参数:
        - station_id_lst: 要检查的站点ID列表，如果为None则检查所有ID
        """
        multi_dataset_ids = self.detect_multi_dataset_ids(station_id_lst)

        if not multi_dataset_ids:
            print("未发现存在于多个数据集中的ID")
            return

        print(f"=== 多数据集ID信息 (共{len(multi_dataset_ids)}个) ===")
        for station_id, datasets in multi_dataset_ids.items():
            print(f"\n站点ID: {station_id}")
            print(f"存在于 {len(datasets)} 个数据集中: {', '.join(datasets)}")

    def auto_get_station_info(
        self,
        station_id_lst: list[str],
        t_range: list = None,
        static_lst: list = None,
        dynamic_list: list = None,
    ) -> dict[str, dict]:
        """
        自动检测并获取站点 ID 列表的信息。

        参数同 get_station_info。

        返回: dict[str, dict] （与 get_station_info 相同）
        """
        return self.get_station_info(
            station_id_lst,
            t_range,
            static_lst,
            dynamic_list,
        )  # 直接调用调整后的方法


# 示例使用函数（更新为支持多数据集）
def print_station_info(
    station_id_lst: list[str] or str,
    t_range: list = None,
    static_lst: list = None,
    dynamic_lst: list = None,
):
    unified = UnifiedCamelsDataset()  # 无需指定类型
    results = unified.auto_get_station_info(
        station_id_lst, t_range, static_lst, dynamic_lst
    )

    for station_id, info in results.items():
        print(f"\n{'='*60}")
        print(f"站点 ID: {station_id}")

        datasets_info = info['datasets']
        dataset_count = len(datasets_info)

        if dataset_count == 1:
            dataset_name = list(datasets_info.keys())[0]
            print(f"来自数据集: {dataset_name}")

            dataset_info = datasets_info[dataset_name]
            print("\n静态属性:")
            print(dataset_info['static_attrs'])

            if dataset_info['timeseries'] is not None:
                print("\n时间序列数据:")
                print(dataset_info['timeseries'])
            else:
                print("\n无时间序列数据（请指定 t_range）")

        else:
            print(
                f"存在于 {dataset_count} 个数据集中: {', '.join(datasets_info.keys())}"
            )

            for dataset_name, dataset_info in datasets_info.items():
                print(f"\n--- {dataset_name} ---")
                print("静态属性:")
                print(dataset_info['static_attrs'])

                if dataset_info['timeseries'] is not None:
                    print("时间序列数据:")
                    print(dataset_info['timeseries'])
                else:
                    print("无时间序列数据（请指定 t_range）")


def print_multi_dataset_stations(
    station_id_lst: list[str] or str,
    t_range: list = None,
    static_lst: list = None,
    dynamic_lst: list = None,
):
    """
    专门用于打印多数据集站点信息的函数。

    参数:
    - station_id_lst: 站点ID列表或单个ID
    - t_range: 时间范围
    - static_lst: 静态属性列表
    - dynamic_lst: 动态变量列表
    """
    unified = UnifiedCamelsDataset()

    # 首先检测哪些是多数据集ID
    if isinstance(station_id_lst, str):
        station_id_lst = [station_id_lst]

    multi_dataset_ids = unified.detect_multi_dataset_ids(station_id_lst)

    if not multi_dataset_ids:
        print("指定的ID中没有存在于多个数据集中的站点")
        return

    print(f"发现 {len(multi_dataset_ids)} 个多数据集站点")

    # 只处理多数据集ID
    results = unified.auto_get_station_info(
        list(multi_dataset_ids.keys()), t_range, static_lst, dynamic_lst
    )

    for station_id, info in results.items():
        print(f"\n{'='*80}")
        print(f"多数据集站点 ID: {station_id}")

        datasets_info = info['datasets']
        print(
            f"存在于 {len(datasets_info)} 个数据集中: {', '.join(datasets_info.keys())}"
        )

        for dataset_name, dataset_info in datasets_info.items():
            print(f"\n{'='*40} {dataset_name} {'='*40}")
            print("静态属性:")
            print(dataset_info['static_attrs'])

            if dataset_info['timeseries'] is not None:
                print("\n时间序列数据:")
                print(dataset_info['timeseries'])
            else:
                print("\n无时间序列数据（请指定 t_range）")


def compare_station_across_datasets(
    station_id: str,
    t_range: list = None,
    static_lst: list = None,
    dynamic_lst: list = None,
):
    """
    比较站点在不同数据集中的信息。

    参数:
    - station_id: 站点ID
    - t_range: 时间范围
    - static_lst: 静态属性列表
    - dynamic_lst: 动态变量列表
    """
    unified = UnifiedCamelsDataset()

    # 检测该ID存在于哪些数据集中
    datasets_for_id = unified._find_datasets_for_id(station_id)

    if len(datasets_for_id) <= 1:
        print(f"站点 {station_id} 只存在于 {len(datasets_for_id)} 个数据集中，无需比较")
        return

    print(f"=== 站点 {station_id} 跨数据集比较 ===")
    print(
        f"存在于 {len(datasets_for_id)} 个数据集中: {', '.join(datasets_for_id.keys())}"
    )

    results = unified.auto_get_station_info(
        [station_id], t_range, static_lst, dynamic_lst
    )

    station_info = results[station_id]
    datasets_info = station_info['datasets']

    # 比较静态属性
    print(f"\n--- 静态属性比较 ---")
    static_comparison = {}
    for dataset_name, dataset_info in datasets_info.items():
        static_df = dataset_info['static_attrs']
        if static_df is not None:
            static_comparison[dataset_name] = static_df
            print(f"\n{dataset_name} 静态属性:")
            print(static_df)

    # 比较时间序列
    if t_range is not None:
        print(f"\n--- 时间序列比较 ---")
        for dataset_name, dataset_info in datasets_info.items():
            timeseries_df = dataset_info['timeseries']
            if timeseries_df is not None:
                print(f"\n{dataset_name} 时间序列:")
                print(timeseries_df)
            else:
                print(f"\n{dataset_name}: 无时间序列数据")
