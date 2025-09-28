from unified_camels import print_station_info

'''
# 输入站点 ID 列表（自动检测类型）
print('the first type:')
print_station_info(
    ['912105A', '01011000'],
    static_lst=['p_mean'],
    dynamic_lst=['q_cms_obs'],
    t_range=['1980-01-01', '1980-01-02'],
)
print('--------------------------------')
print('the second type:')
print_station_info(
    ['912105A', '01011000'],
    dynamic_lst=['q_cms_obs'],
    t_range=['1980-01-01', '1980-01-02'],
)
print('--------------------------------')
print('the third type:')
print_station_info(
    ['912105A', '01011000'],
    static_lst=['p_mean'],
    t_range=['1980-01-01', '1980-01-02'],
)
print('--------------------------------')
print('the fourth type:')
print_station_info(
    ['912105A', '01011000'],
    t_range=['1980-01-01', '1980-01-02'],
)
print('--------------------------------')
print('the fifth type:')
print_station_info(
    ['912105A', '01011000'],
    static_lst=['p_mean'],
)
print('--------------------------------')
print('the sixth type:')
print_station_info(
    ['912105A', '01011000'],
)
print('--------------------------------')
print('the seventh type:')
print_station_info(
    ['912105A', '01011000'],
    static_lst=['p_mean'],
    dynamic_lst=['q_cms_obs'],
)
print('--------------------------------')
print('the eighth type:')
print_station_info(
    ['912105A', '01011000'],
    dynamic_lst=['q_cms_obs'],
)
'''
print_station_info(
    [
        '1083',
    ],
)
