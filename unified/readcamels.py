from unified.unified_camels import print_station_info


# 输入站点 ID 列表（自动检测类型）
print_station_info(
    ['912105A', '01011000'],
    static_lst=['p_mean'],
    dynamic_lst=['q_cms_obs'],
    t_range=['1980-01-01', '1980-01-02'],
)
