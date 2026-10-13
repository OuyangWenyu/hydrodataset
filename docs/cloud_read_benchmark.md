# 本地与云端数据读取性能测试说明

本文整理 hydrodataset 项目中 CAMELS-US 与 CAMELSH 的本地/云端读取测试，说明历史测试结果、计时边界、存储访问方式、相关脚本及复现条件。


## 1. 测试目标与术语

测试主要回答两个问题：

1. 对一个流域、一个变量和一段较短时间范围进行按需访问时，本地 NC、云端 NC 和云端 Zarr 的延迟分别是多少？
2. 全量读取 CAMELS-US 时，不同时间 chunk 大小如何影响对象数量、读取耗时和逻辑吞吐率？

本文中的 `ECS→OSS` 表示测试程序运行在阿里云 ECS 上，由 ECS 向 OSS 发起读取请求。若按数据返回方向描述，则数据实际由 OSS 传输到 ECS。历史云端测试使用北京 ECS 和北京 OSS 内网 endpoint。

`cold` 表示第一次执行读取；`warm` 表示同一进程中后续重复执行。warm 结果可能受 HTTP 连接复用、操作系统缓存、客户端缓存和 OSS 侧状态影响，不等同于严格清空所有缓存后的独立实验。

## 2. 历史测试环境

| 项目 | 本地 | 云端 |
| --- | --- | --- |
| 设备 | Windows PC（SSD） | 北京 ECS |
| 数据位置 | 本地 NetCDF 缓存 | 阿里云 OSS |
| 网络 | 本地文件系统 | 阿里云 OSS 内网 |
| CAMELS-US Zarr chunk | — | `(100, 365)`，约 0.3 MB/chunk |
| CAMELSH Zarr chunk | — | `(100, 8760)`，约 3.5 MB/chunk |

上表是按需读取对比发生时的历史配置。语雀后文还记录了 CAMELSH 后续生成方案将流域 chunk 调整为 400；这是另一个阶段的方案，不应与表中的 `(100, 8760)` 混为同一次测试。

## 3. 历史按需读取结果

### 3.1 CAMELS-US

| 操作 | 本地 NC（SSD） | 云端 NC（ECS→OSS） | 云端 Zarr（ECS→OSS） | NC 倍数 | Zarr 倍数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 打开属性（0.6 MB） | 0.069s | 0.248s | **0.315s** | ×3.6 | ×4.6 |
| 读 `p_mean` | 0.000s | 0.001s | 0.179s | — | — |
| 打开时序（171 MB） | 0.373s | 3.462s | **0.066s** | ×9.3 | **×0.18** |
| 读 `streamflow` | 0.001s | 0.002s | 0.029s | — | — |
| **总计** | **0.443s** | **3.713s** | **0.589s** | ×8.4 | ×1.3 |

测试条件：

- 流域：`01013500`
- 静态属性：`p_mean`
- 动态变量：`streamflow`（云端缓存中可能使用原始名 `q_cms_obs`）
- 时间范围：`1990-01-01` 至 `1990-12-31`
- 每项执行 3 次，表中记录平均时间

“打开”和“读取”是两个独立计时步骤：

- “打开属性/时序”记录打开数据集并建立可查询数据结构的时间。
- “读 `p_mean`”记录从已打开的数据集中读取一个流域的一个静态属性值的时间。
- “读 `streamflow`”记录从已打开的数据集中选择一个流域的一年流量并计算平均值的时间。
- “总计”是四个步骤平均耗时的直接相加，例如 `0.315 + 0.179 + 0.066 + 0.029 = 0.589s`。

### 3.2 CAMELSH

| 操作 | 本地 NC（单批） | 本地 `read_ts`（全站） | 云端 NC（单批） | 云端 Zarr（全站，按需） | vs 本地单批 | vs 本地全站 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 打开属性（17 MB） | 0.303s | — | 2.560s | 1.349s | ×4.5 | — |
| 读 `p_mean` | 0.001s | — | 0.002s | 0.063s | — | — |
| 打开时序 | 0.397s | **0.842s** | 11.146s | **0.195s** | **×0.5** | **×0.23** |
| 读 `streamflow` | 0.003s | — | 0.004s | 0.031s | — | — |
| **总计** | **0.704s** | **约 1.2s** | **13.712s** | **1.638s** | ×2.3 | ×1.4 |

测试条件与 CAMELS-US 相同：流域 `01013500`、属性 `p_mean`、1990 年 `streamflow`，每项执行 3 次取平均。

四列含义如下：

- **本地 NC（单批）**：直接打开 `batch001_camelsh_timeseries.nc`。该文件包含 100 个站点，其中包括目标站点。
- **本地 `read_ts`（全站）**：调用 `Camelsh.read_ts_xrdataset()`；接口需要检查全部 58 个 batch 文件，找到包含目标站点的批次。
- **云端 NC（单批）**：从 OSS 下载属性 NC 和单个 batch NC 的完整对象，再从内存打开。
- **云端 Zarr（全站，按需）**：打开覆盖全站的 Zarr 数据集，只读取目标查询涉及的 chunk（历史记录约 3.5 MB）。

语雀中的“Zarr 比 NC 快 52 倍/57 倍”描述的是这种单站按需访问场景中，Zarr 元数据打开相对于“先下载完整云端 NC”的优势，不代表全量数据读取在所有情况下都能获得相同倍数。

## 4. CAMELS-US chunk 全量吞吐测试

语雀还记录了另一组 CAMELS-US 全量读取实验。该实验覆盖全流域、8 个动态变量和完整时间范围，通过改变时间维 chunk 大小观察对象数量与吞吐率。

| 时间 chunk | 时间块数 | 对象数 | cold | warm 最短 | 逻辑吞吐率（warm） |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 365（1 年） | 35 | 280 | 2.893s | 2.183s | 240 MiB/s |
| 1825（5 年） | 8 | 64 | 1.418s | 1.135s | 461 MiB/s |
| 3650（10 年） | 4 | 32 | 1.239s | 0.911s | 575 MiB/s |
| 12784（全时段） | 1 | 8 | 1.186s | 0.874s | 599 MiB/s |

对象数由“时间块数 × 8 个变量”得到。例如时间 chunk 为 365 时，需要访问 `35 × 8 = 280` 个数据对象。全量读取时，增大时间 chunk 能减少 OSS 请求次数，因此吞吐率明显提高；从 3650 增大到 12784 后，收益已开始趋缓。

该结论只适用于全量或大范围顺序读取。对于单流域、短时间范围的随机读取，chunk 过大会产生读取放大：为了获得很小的逻辑切片，仍需下载并解码覆盖该切片的整个 chunk。因此 chunk 需要根据训练批次和典型查询范围折中选择。

表中的 MiB/s 是逻辑吞吐率：

```text
逻辑吞吐率 = 最终数组的 nbytes / 读取耗时
```

它不是 OSS 实际传输字节数除以时间。压缩、chunk 读取放大和元数据请求都会使实际网络流量与逻辑数组大小不同。

## 5. 三种底层读取方式

### 5.1 本地 NetCDF

本地模式通过 `xarray.open_dataset()` 打开 SSD 上的 NetCDF 缓存。打开阶段主要读取文件头、维度、坐标和变量元数据；真正访问 `.values`、`.load()` 或执行聚合时，底层后端才读取对应数组数据。

### 5.2 云端 NetCDF

历史对比脚本没有对远端 NetCDF 做 Range 请求，而是使用 `s3fs.cat_file()` 下载完整对象，再通过 `BytesIO + h5netcdf` 打开：

```text
ECS 发起 OSS 请求
  → 下载完整 NetCDF 对象到 ECS 内存
  → xarray/h5netcdf 打开 BytesIO
  → 选择目标属性或时序
```

因此“打开云端 NC”的耗时包含完整对象下载时间。这是历史脚本的实现选择，不应泛化为所有 NetCDF 远程读取实现都必须如此。

### 5.3 云端 Zarr

历史延迟脚本使用 `xarray` 的 Zarr 引擎，并通过 `storage_options` 将 `hydro_setting.yml` 中的 OSS 配置交给 s3fs/fsspec：

```python
xr.open_dataset(
    path,
    engine="zarr",
    storage_options=s3_config,
    consolidated=False,
    chunks={},
)
```

完整流程如下：脚本运行在 ECS 上，先读取 `hydro_setting.yml` 中的 bucket、访问密钥和 endpoint。xarray 使用 Zarr 引擎打开 OSS 中的数据集，首先读取维度、变量、坐标和 chunk 布局等元数据，建立延迟加载的数据结构。查询指定流域、变量和时间范围后，Zarr 根据索引确定相交的 chunks，s3fs/fsspec 只向 OSS 请求这些对象。对象下载完成后，Dask 负责解码、构造数组并执行平均值等计算。

历史脚本为复现实验使用 `consolidated=False`，因此需要读取多个独立元数据对象。项目正式读取接口则通过 `_open_zarr_pref_consolidated()` 优先尝试 consolidated metadata，以减少元数据 GET 请求；失败时才回退到非 consolidated 模式。

## 6. 测试脚本

| 脚本 | 数据集 | 用途 | 计时粒度 |
| --- | --- | --- | --- |
| `scripts/bench_cloud.py` | CAMELS-US | 恢复历史单站延迟对比 | 打开属性、读一个属性、打开时序、读一年流量分别计时 |
| `scripts/bench_camelsh_cloud.py` | CAMELSH | 根据语雀记录重构历史四模式对比 | 与历史表格相同的分步骤计时 |
| `scripts/bench_read_speed.py` | CAMELS-US | 当前正式接口的批量/全量读取测速 | `read_*_xrdataset(...).load()` 端到端计时 |
| `scripts/bench_camelsh_read.py` | CAMELSH | 当前正式接口的可调批量读取测速 | `read_*_xrdataset(...).load()` 端到端计时 |

### 6.1 `bench_cloud.py`

该脚本来自 Git 历史中的 CAMELS-US 对比脚本，现已改为从 `~/hydro_setting.yml` 读取 OSS 配置，同时保留环境变量覆盖。

模式：

```text
local  → 本地 CAMELS-US NetCDF
cloud  → OSS 上的完整 CAMELS-US NetCDF
zarr   → OSS 上的 CAMELS-US Zarr
```

默认查询固定为流域 `01013500`、`p_mean`、1990 年 `streamflow`，每项重复 3 次取平均。`cloud` 模式要求远端目录中实际存在 `.nc` 文件；当前项目通常只在 `zarr/` 下保存 Zarr，需要通过 `CLOUD_CACHE_PATH` 指定旧 NC 所在位置。

### 6.2 `bench_camelsh_cloud.py`

该脚本根据语雀表格和 CAMELSH 当前缓存结构重构，提供四种模式：

```text
local-batch → 直接打开本地 batch001
local-api   → 调用 Camelsh.read_ts_xrdataset() 扫描本地批文件
cloud-nc    → 从 OSS 下载完整属性 NC 和 batch001 NC
cloud-zarr  → 从 OSS Zarr 按需读取
```

默认云端 Zarr 根路径由 `storage.s3.bucket + prefix + /zarr` 组成。语雀没有保存旧云端 NC 的确切对象路径，脚本暂用历史推定值 `s3://camelsh/cache/`；可通过 `CAMELSH_CLOUD_NC_PATH` 覆盖。

### 6.3 `bench_read_speed.py`

该脚本测试 CAMELS-US 正式读取路径：

```python
ds.read_ts_xrdataset(...).load()
ds.read_attr_xrdataset(...).load()
```

默认读取 100 个流域、5 年、8 个动态变量和 17 个静态属性。第一次执行记为 cold，后续 `--repeats` 次记为 warm。每次调用都会重新进入读取接口并通过 `.load()` 强制物化结果。

全量复现命令：

```powershell
uv run python scripts/bench_read_speed.py local --basins 671 --years 35 --repeats 3
```

```bash
uv run python scripts/bench_read_speed.py cloud --basins 671 --years 35 --repeats 3
```

本地模式读取 NetCDF 缓存，云端模式通过正式接口读取 OSS Zarr。两者结果代表应用实际等待时间，但底层存储格式不同，因此不是相同格式的纯硬件对比。

### 6.4 `bench_camelsh_read.py`

该脚本测试 CAMELSH 正式接口的批量读取。默认读取 1000 个流域、10 年小时数据、8 个动态变量和 5 个静态属性，也分别报告 cold、warm 与逻辑吞吐率。

示例：

```powershell
uv run python scripts/bench_camelsh_read.py local
```

```bash
uv run python scripts/bench_camelsh_read.py cloud
uv run python scripts/bench_camelsh_read.py cloud --basins 200 --years 2
uv run python scripts/bench_camelsh_read.py cloud --vars streamflow,precipitation --attrs all
```

## 7. 配置要求

四个脚本最终都应使用当前用户目录下的 `~/hydro_setting.yml`。示例中必须使用占位符，不应把真实密钥提交到仓库：

```yaml
storage:
  default_source: local
  local:
    root: /path/to/datasets
  cache: /path/to/netcdf-cache
  s3:
    bucket: hydrodataset
    prefix: ""
    access_key_id: YOUR_ACCESS_KEY_ID
    secret_access_key: YOUR_SECRET_ACCESS_KEY
    endpoint_url: https://oss-cn-beijing-internal.aliyuncs.com
```

endpoint 选择：

- 北京 ECS 访问北京 OSS：使用 `https://oss-cn-beijing-internal.aliyuncs.com`。
- 本地开发机访问 OSS：使用 `https://oss-cn-beijing.aliyuncs.com`。

历史延迟脚本还允许环境变量覆盖 YAML，以便指定旧对象路径或临时测试环境。环境变量优先级高于 YAML，但不得将真实密钥写入脚本、文档或提交记录。

## 8. 复现和比较要求

只有以下条件一致时，结果才适合直接比较：

1. 数据集版本、流域数量、变量数量和时间范围一致。
2. Zarr shape、dtype、压缩器和 chunk 布局一致。
3. 使用相同 endpoint，并确认 ECS 与 OSS 位于同一地域且走内网。
4. Python、xarray、Dask、Zarr、s3fs/fsspec 和 aiobotocore 版本一致。
5. 计时边界一致：不能把“仅打开元数据”与“`.load()` 完整加载数据”直接比较。
6. cold/warm 定义、重复次数和汇总方法一致。
7. 吞吐率使用同一种口径，并明确是逻辑数组大小还是实际网络字节数。

当前 OSS 的 CAMELS-US Zarr 已与历史测试阶段不同；例如当前读取时可能显示不同的 basin 数量和 `(basin, time)` chunks。因此，历史表格适合作为设计依据和数量级参考，而不是当前环境必须逐项达到的固定验收值。

## 9. 结论

- 对单站、单变量、短时间范围的查询，Zarr 可以避免下载完整 NC 文件，显著降低云端按需访问延迟。
- 对全量读取，增大时间 chunk 能减少 OSS 对象请求数量并提高吞吐，但达到一定大小后收益趋缓。
- chunk 越大并不意味着所有查询都越快；随机小范围读取需要考虑读取放大。
- 历史延迟脚本适合解释存储格式差异，当前批量脚本更接近训练和数据处理程序的真实端到端读取路径。
- 比较测试结果前，必须先确认 endpoint、数据布局、依赖版本和计时边界一致。
