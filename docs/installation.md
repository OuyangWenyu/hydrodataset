# Installation

We strongly recommend using a virtual environment to manage dependencies and avoid package conflicts.

## Requirements

- **Python**: 3.10 or higher
- **Operating System**: Windows, Linux, or macOS
- **Dependencies**: Automatically installed with pip (xarray, netCDF4, pandas, numpy, pint, AquaFetch, etc.)

## For Users

### Using uv (Recommended)

We recommend using [uv](https://github.com/astral-sh/uv) for fast, reliable package and environment management:

```bash
# Install uv if you haven't already
pip install uv

# Install hydrodataset with uv
uv pip install hydrodataset
```

This installs the latest stable release along with all required dependencies, significantly faster than traditional pip.

### Using pip (Alternative)

If you prefer traditional pip:

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install hydrodataset
pip install hydrodataset
```

### Using conda

If you prefer conda, you can install from conda-forge:

```bash
# Create a new conda environment
conda create -n hydro python=3.10
conda activate hydro

# Install from conda-forge
conda install -c conda-forge hydrodataset
```

### Verify Installation

After installation, verify it works:

```python
python -c "import hydrodataset; print(hydrodataset.__version__)"
```

## For Developers

If you want to contribute to hydrodataset or modify the source code, follow these steps:

### Using uv (Recommended)

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable package and environment management:

```bash
# Clone the repository
git clone https://github.com/OuyangWenyu/hydrodataset.git
cd hydrodataset

# Install uv if you haven't already
pip install uv

# Create virtual environment and install all dependencies
uv sync --all-extras
```

The `--all-extras` flag installs:
- Base dependencies (required for core functionality)
- Development tools (pytest, black, flake8, etc.)
- Documentation tools (mkdocs, mkdocstrings, etc.)

### Using pip (Alternative)

If you prefer traditional pip:

```bash
# Clone the repository
git clone https://github.com/OuyangWenyu/hydrodataset.git
cd hydrodataset

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with all extras
pip install -e ".[dev,docs,lint]"
```

### Verify Development Installation

```bash
# Run tests
pytest

# Check code formatting
black hydrodataset tests

# Run linting
flake8 hydrodataset tests

# Build documentation
mkdocs serve
```

## Post-Installation Setup

### Create Configuration File

After installation, create a `hydro_setting.yml` file in your **home directory**:

**Windows**: `C:\Users\YourUsername\hydro_setting.yml`
**Linux/Mac**: `~/hydro_setting.yml`

**Content (local only):**
```yaml
storage:
  local:
    root: D:/data/hydrodatasets   # root directory that contains all dataset folders
  cache: cache                    # relative to local.root, or supply an absolute path
```

**For cloud access, add the `s3` block** (used by `source="cloud"`):
```yaml
storage:
  s3:
    bucket: hydrodataset          # required for cloud access
    prefix: ""                    # optional prefix inside the bucket
    endpoint_url: https://oss-cn-beijing.aliyuncs.com
    access_key_id: <your-access-key>
    secret_access_key: <your-secret-key>
```

`storage.s3.*` contains credentials — never commit it to a repository.

**Important**: Update the paths according to your system. Ensure:
- The `root` directory already exists — local path resolution (`source="local"`) fails if it does not
- You have write permissions
- Sufficient disk space (raw data + NetCDF cache can be several GB per dataset)

### Download Data

hydrodataset uses [AquaFetch](https://github.com/hyex-research/AquaFetch) to fetch raw data. Some datasets download automatically, while others require manual download. Check the AquaFetch documentation for dataset-specific instructions.

## Troubleshooting

### pip installation fails

If you encounter errors during installation:

```bash
# Upgrade pip, setuptools, and wheel
pip install --upgrade pip setuptools wheel

# Try installing again
pip install hydrodataset
```

### Import errors after installation

```bash
# Ensure you're in the correct environment
which python  # Should point to your virtual environment

# Reinstall
pip uninstall hydrodataset
pip install hydrodataset
```

### AquaFetch dependency issues

hydrodataset depends on the development version of AquaFetch. If you encounter issues:

```bash
# Install AquaFetch directly from GitHub
pip install git+https://github.com/hyex-research/AquaFetch.git@dev
```

### Configuration file not found

Error: `FileNotFoundError: hydro_setting.yml not found`

**Solution**: Ensure `hydro_setting.yml` is in your home directory:
```bash
# Check home directory
echo $HOME  # Linux/Mac
echo %USERPROFILE%  # Windows

# Create file
touch ~/hydro_setting.yml  # Linux/Mac
type nul > %USERPROFILE%\hydro_setting.yml  # Windows
```

## Upgrading

### Upgrade to Latest Version

```bash
pip install --upgrade hydrodataset
```

### Upgrade from conda

```bash
conda update -c conda-forge hydrodataset
```

## Uninstallation

```bash
# Using pip
pip uninstall hydrodataset

# Using conda
conda remove hydrodataset
```

## Next Steps

After installation:
1. ✅ Create `hydro_setting.yml` configuration file
2. 📖 Read the [Usage Guide](usage.md)
3. 🚀 Try the [Quick Start](index.md#quick-start) examples
4. 📚 Browse the [API Documentation](api/hydrodataset.md)

If you encounter issues, check the [FAQ](faq.md) or open an issue on [GitHub](https://github.com/OuyangWenyu/hydrodataset/issues).
