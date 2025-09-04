from hydrodataset import SETTING, CACHE_DIR
from hydrodataset.camelsh import Camelsh  # 从 camelsh.py 导入
from hydrodataset.camels_aus_aqua import CamelsAus  # 从 camels_aus_aqua.py 导入
import pandas as pd
import xarray as xr


class UnifiedCamelsDataset:
    # 类属性：缓存站点列表（初始化时加载一次）
    _camelsh_stations = None
    _camels_aus_stations = None

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

    def get_station_info(
        self,
        station_id_lst: list[str],
        t_range: list = None,
        static_lst: list = None,
        dynamic_lst: list = None,
    ) -> dict[str, dict]:
        """
        根据站点 ID 列表 获取信息（支持多个 ID）。

        参数:
        - station_id_lst: 站点 ID 列表 (list[str]) 或单个 str
        - t_range: 时间范围 (list)，可选
        - var_lst: 变量列表 (list)，可选

        返回:
        - dict: {station_id: {'static_attrs': DataFrame, 'timeseries': DataFrame or None}}
        """
        if isinstance(station_id_lst, str):  # 支持单个 ID（转换为列表）
            station_id_lst = [station_id_lst]

        results = {}  # 结果字典: {id: info}

        for station_id in station_id_lst:
            try:
                # 确定数据集类型（内部检测）
                if station_id in UnifiedCamelsDataset._camelsh_stations:
                    self.dataset = Camelsh(self.data_path)
                elif station_id in UnifiedCamelsDataset._camels_aus_stations:
                    self.dataset = CamelsAus(self.data_path)
                else:
                    print(f"警告: 跳过无效站点 ID '{station_id}'（不属于任何数据集）。")
                    continue

                info = {}

                # 获取静态属性
                attrs_ds = self.dataset.read_attr_xrdataset(
                    gage_id_lst=[station_id], var_lst=static_lst
                )
                info['static_attrs'] = attrs_ds.to_dataframe().reset_index()

                # 获取时间序列

                ts_ds = self.dataset.read_ts_xrdataset(
                    gage_id_lst=[station_id], t_range=t_range, var_lst=dynamic_lst
                )
                info['timeseries'] = ts_ds.to_dataframe().reset_index()

                results[station_id] = info
            except Exception as e:
                print(f"错误: 处理站点 '{station_id}' 失败 - {e}")

        return results

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


# 示例使用函数（更新为支持列表）
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
        print(f"\n站点 ID: {station_id}")

        print("静态属性:")
        print(info['static_attrs'])

        if info['timeseries'] is not None:
            print("时间序列数据:")
            print(info['timeseries'])
        else:
            print("无时间序列数据（请指定 t_range）")
