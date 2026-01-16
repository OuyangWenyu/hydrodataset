import netCDF4 as nc
import numpy as np
import sys

# 设置输出编码为UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# File path
file_path = "D:/netcdf/camels_us_timeseries.nc"

# Open NetCDF file
try:
    dataset = nc.Dataset(file_path, 'r')
except FileNotFoundError:
    print(f"错误: 文件未找到 {file_path}")
    sys.exit(1)

print("=" * 80)
print(f"NetCDF 文件属性信息: {file_path}")
print("=" * 80)

# 1. Global Attributes (全局属性)
print("\n" + "=" * 80)
print("全局属性 (Global Attributes):")
print("=" * 80)
if dataset.ncattrs():
    for attr_name in dataset.ncattrs():
        attr_value = getattr(dataset, attr_name)
        print(f"  {attr_name}: {attr_value}")
else:
    print("  无全局属性")

# 2. Dimensions (维度)
print("\n" + "=" * 80)
print("维度 (Dimensions):")
print("=" * 80)
for dim_name, dim in dataset.dimensions.items():
    unlimited = "(UNLIMITED)" if dim.isunlimited() else ""
    print(f"  {dim_name}: {len(dim)} {unlimited}")

# 3. Variables (变量及其属性)
print("\n" + "=" * 80)
print("变量及其属性 (Variables and Attributes):")
print("=" * 80)
for var_name in dataset.variables.keys():
    var = dataset.variables[var_name]
    print(f"\n变量: {var_name}")
    print(f"  维度: {var.dimensions}")
    print(f"  形状: {var.shape}")
    print(f"  数据类型: {var.dtype}")

    # Variable attributes
    if var.ncattrs():
        print(f"  属性:")
        for attr_name in var.ncattrs():
            attr_value = getattr(var, attr_name)
            print(f"    {attr_name}: {attr_value}")
    else:
        print(f"  属性: 无")

print("\n" + "=" * 80)

# Close file
dataset.close()
print("\n文件已关闭")