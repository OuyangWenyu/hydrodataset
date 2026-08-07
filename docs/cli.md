# hydrodataset 命令行(CLI)化

本文记录把 hydrodataset 从"只能作为 Python 库调用"扩展为"带命令行工具"的调研、设计与实现过程。

## 一、背景与目标

项目已有清晰的公共 API(`resolve_data_path` / `open_dataset` + 各 reader 方法),但没有命令行入口。为了方便快速查询/读取数据、便于脚本与智能体调用,给项目增加一个统一命令 `hydrodataset`。

原则:**CLI 是薄薄一层**——只做"解析参数 → 调用现有 API → 格式化输出",不包含任何数据处理逻辑。

## 二、调研:项目现状

| 方面 | 现状 |
|------|------|
| 打包后端 | `pyproject.toml` + hatchling(`hatchling.build`),无 `[project.scripts]` 入口 |
| 公共 API | `resolve_data_path(dataset_id, source=...)`、`open_dataset(dataset_id, source=...)`(返回实例化好的 reader)、`READER_ALIASES`、`_DEFAULT_REGISTRY`、`ResolverContext`、settings 助手 |
| reader 方法 | `read_object_ids` / `read_ts_xrdataset` / `read_attr_xrdataset` / `available_static_features` / `available_dynamic_features` / `default_t_range` / `cache_*_to_zarr` |
| 已有类 CLI | `examples/read_dataset.py`(argparse 手写的单数据集读取示例)——作为设计参考,但**不改造它**,CLI 另起独立模块 |

关键结论:`open_dataset(dataset_id, source=...)` 已经是理想的 CLI 主干——CLI 只需把命令行参数转成对它和 reader 方法的调用。

## 三、设计决策

1. **框架:Typer**(基于 click)。类型注解即参数、自动生成 help、自带 shell 补全、样板少;底层 click 本就是间接依赖。作为**主依赖**加入(轻量,保证命令开箱即用)。
2. **命令名:`hydrodataset`**(清晰,不加短别名)。
3. **不迁移 `examples/read_dataset.py`**:它已相对成熟,保留为独立示例,与 CLI 做区分。
4. **不做 `cache` 命令(v1)**:读属性/时序时若缓存缺失会**自动触发生成**,所以纯取数不需要它;其额外价值仅为"主动预生成 / 强制重建",留待 v2。
5. **`--source` 语义与库一致**:不传则回退配置里的 `storage.default_source`(库侧已实现该回退)。

## 四、实现

### 4.1 依赖与入口

`pyproject.toml`:
```toml
[project]
dependencies = [ ..., "typer" ]

[project.scripts]
hydrodataset = "hydrodataset.cli:app"
```
- `typer` 通过 `uv add typer` 加入(同时更新 `uv.lock`)。
- 入口指向 Typer 实例 `app`(Typer 实例可调用,`app()` 即运行 CLI)。安装后即可用 `hydrodataset` 命令,或 `python -m hydrodataset.cli`。

### 4.2 模块 `hydrodataset/cli.py`

薄层结构,核心约定:
- 每个命令内部**惰性导入**库函数(避免 CLI 启动时的重导入开销)。
- **`--source / -s`** 为共享选项(`Optional[str]`,默认 `None` → 走 `default_source`)。
- 辅助:`_split()` 把逗号串转 list;`_fail()` 打印红色错误 + 非零退出码;`_write_or_show()` 把 `xr.Dataset` 写 `.nc` / `.csv` 或打印摘要。
- 错误处理:`DatasetResolutionError`、缺变量、缺数据等 → 友好提示 + 退出码 1,不甩堆栈。

## 五、命令一览(v1)

| 命令 | 作用 | 底层调用 |
|------|------|---------|
| `list` | 列出注册表所有数据集(id / reader / module.class) | `_DEFAULT_REGISTRY` + `READER_ALIASES` |
| `resolve <ds> [--source]` | 打印解析出的路径 / S3 URI | `resolve_data_path` |
| `info <ds> [--source]` | 显示路径、时间范围、站点数、静/动态属性清单 | `open_dataset` + `available_*` / `default_t_range` |
| `ids <ds> [--source] [--limit N]` | 列出站点 ID | `read_object_ids` |
| `read-ts <ds> [--gages] [--vars] [--t-range] [-o out]` | 读时序,输出 NC/CSV/终端 | `read_ts_xrdataset` |
| `read-attr <ds> [--gages] [--vars] [-o out]` | 读静态属性 | `read_attr_xrdataset` |
| `config` | 打印生效配置(default_source / local.root / cache / s3;**密钥打码**) | settings 助手 |

选项约定:
- `--gages/-g`:逗号分隔站点 id,省略则全部。
- `--vars/-v`:逗号分隔标准变量名,省略则全部(时序=全部动态、属性=全部静态)。
- `--t-range/-t`:`START,END`,省略则数据集默认范围。
- `--out/-o`:`.nc`(to_netcdf)/ `.csv`(to_dataframe().to_csv);省略则打印摘要。

## 六、安装与使用

### 本地(uv)
```bash
uv add typer         # 或 uv sync（pyproject 已含 typer）
uv run hydrodataset --help
```

### 服务器(无 uv 于 PATH 时也可)
```bash
cd /root/hydrodataset
uv sync                                   # 装 typer + 注册入口
uv run hydrodataset --help
# 或仅装 typer 后用模块方式：
uv pip install typer
uv run python -m hydrodataset.cli --help
```

### 示例
```bash
hydrodataset list
hydrodataset config
hydrodataset info camels_us --source cloud
hydrodataset ids camels_us --limit 10
hydrodataset resolve camels_us --source cloud            # -> s3://hydrodataset/
hydrodataset read-attr camels_us --gages 01013500 --vars area,p_mean -o attr.csv
hydrodataset read-ts bull --source cloud --gages BULL_10004 --vars precipitation -o ts.nc
```

`--source` 不传时按配置 `default_source`;显式传 `--source local|cloud` 覆盖。

## 七、后续规划(v2+)

- `cache <ds> [--attrs] [--ts] [--force] [--source]`:主动预生成云端 zarr / 本地 NC,`--force` 封装"删旧缓存 → 重建"(替代手动 `fs.rm` + 重跑)。
- `multi read --datasets a,b --gages ...`:基于 `MultiDatasetReader` 的跨数据集读取。
- `--config <yml>`:指定非默认的 `hydro_setting.yml`。
- shell 补全安装说明(`hydrodataset --install-completion`)、进度条。

## 八、涉及文件

- 新增 `hydrodataset/cli.py`
- 修改 `pyproject.toml`(`typer` 依赖 + `[project.scripts]` 入口)、`uv.lock`
- **未改** `examples/read_dataset.py`(保持独立示例)
