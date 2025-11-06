# Standard Variables

## Overview

The `StandardVariable` class defines standardized variable names used across all datasets in hydrodataset. This ensures consistency when accessing data from different hydrological datasets, even when the underlying dataset uses different naming conventions.

## Purpose

Different hydrological datasets use various naming conventions for the same variables:
- CAMELS-US might use "dayl" for daylight duration
- CAMELS-AUS might use "solarrad_AWAP" for solar radiation
- CARAVAN might use "temperature_2m_mean" for mean temperature

StandardVariable provides a unified naming system so users can request `StandardVariable.TEMPERATURE_MEAN` regardless of the underlying dataset.

## Variable Categories

### Streamflow
- **`STREAMFLOW`**: River discharge/streamflow

### Precipitation
- **`PRECIPITATION`**: Precipitation/rainfall

### Temperature
- **`TEMPERATURE_MAX`**: Maximum air temperature
- **`TEMPERATURE_MIN`**: Minimum air temperature
- **`TEMPERATURE_MEAN`**: Mean air temperature

### Evapotranspiration
- **`POTENTIAL_EVAPOTRANSPIRATION`**: Potential evapotranspiration (PET)
- **`EVAPOTRANSPIRATION`**: Actual evapotranspiration (ET/AET)
- **`EVAPORATION`**: Evaporation from water surfaces

### Radiation
- **`SOLAR_RADIATION`**: Solar/shortwave radiation
- **`SOLAR_RADIATION_MIN`**: Minimum solar radiation
- **`SOLAR_RADIATION_MAX`**: Maximum solar radiation
- **`THERMAL_RADIATION`**: Thermal/longwave radiation
- **`THERMAL_RADIATION_MIN`**: Minimum thermal radiation
- **`THERMAL_RADIATION_MAX`**: Maximum thermal radiation

### Snow
- **`SNOW_WATER_EQUIVALENT`**: Snow water equivalent
- **`SNOW_WATER_EQUIVALENT_MIN`**: Minimum snow water equivalent
- **`SNOW_WATER_EQUIVALENT_MAX`**: Maximum snow water equivalent
- **`SNOW_DEPTH`**: Snow depth

### Wind
- **`WIND_SPEED`**: Wind speed
- **`U_WIND_SPEED`**: U-component of wind speed
- **`U_WIND_SPEED_MIN`**: Minimum U-component wind speed
- **`U_WIND_SPEED_MAX`**: Maximum U-component wind speed
- **`V_WIND_SPEED`**: V-component of wind speed
- **`V_WIND_SPEED_MIN`**: Minimum V-component wind speed
- **`V_WIND_SPEED_MAX`**: Maximum V-component wind speed

### Atmospheric
- **`VAPOR_PRESSURE`**: Vapor pressure
- **`SPECIFIC_HUMIDITY`**: Specific humidity
- **`RELATIVE_HUMIDITY`**: Relative humidity
- **`SURFACE_PRESSURE`**: Surface atmospheric pressure
- **`SURFACE_PRESSURE_MIN`**: Minimum surface pressure
- **`SURFACE_PRESSURE_MAX`**: Maximum surface pressure

### Soil
- **`SOIL_MOISTURE`**: Soil moisture (general)
- **`VOLUMETRIC_SOIL_WATER_LAYER1`**: Volumetric soil water content, layer 1
- **`VOLUMETRIC_SOIL_WATER_LAYER1_MIN`**: Minimum soil water, layer 1
- **`VOLUMETRIC_SOIL_WATER_LAYER1_MAX`**: Maximum soil water, layer 1
- **`VOLUMETRIC_SOIL_WATER_LAYER2`**: Volumetric soil water content, layer 2
- **`VOLUMETRIC_SOIL_WATER_LAYER2_MIN`**: Minimum soil water, layer 2
- **`VOLUMETRIC_SOIL_WATER_LAYER2_MAX`**: Maximum soil water, layer 2
- **`VOLUMETRIC_SOIL_WATER_LAYER3`**: Volumetric soil water content, layer 3
- **`VOLUMETRIC_SOIL_WATER_LAYER3_MIN`**: Minimum soil water, layer 3
- **`VOLUMETRIC_SOIL_WATER_LAYER3_MAX`**: Maximum soil water, layer 3
- **`VOLUMETRIC_SOIL_WATER_LAYER4`**: Volumetric soil water content, layer 4
- **`VOLUMETRIC_SOIL_WATER_LAYER4_MIN`**: Minimum soil water, layer 4
- **`VOLUMETRIC_SOIL_WATER_LAYER4_MAX`**: Maximum soil water, layer 4

### Daylight
- **`DAYLIGHT_DURATION`**: Hours of daylight

## Usage

### Basic Usage

```python
from hydrodataset import StandardVariable
from hydrodataset.camels_us import CamelsUs

ds = CamelsUs(data_path)

# Request data using standard variable names
timeseries = ds.read_ts_xrdataset(
    gage_id_lst=basin_ids,
    t_range=["1990-01-01", "1995-12-31"],
    var_lst=[
        StandardVariable.STREAMFLOW,
        StandardVariable.PRECIPITATION,
        StandardVariable.TEMPERATURE_MEAN
    ]
)
```

### Using String Names

You can also use lowercase string versions of the standard variable names:

```python
# These are equivalent
timeseries = ds.read_ts_xrdataset(
    gage_id_lst=basin_ids,
    t_range=["1990-01-01", "1995-12-31"],
    var_lst=["streamflow", "precipitation", "temperature_mean"]
)
```

### Multiple Data Sources

Some datasets provide the same variable from multiple sources. You can specify the source:

```python
# Request precipitation from ERA5-Land source instead of default
timeseries = ds.read_ts_xrdataset(
    gage_id_lst=basin_ids,
    t_range=["1990-01-01", "1995-12-31"],
    var_lst=[
        ("precipitation", "era5land"),  # Specify source
        "streamflow"  # Use default source
    ]
)
```

### Checking Available Variables

```python
# List all available standard variables for a dataset
print(ds.available_dynamic_features)
print(ds.available_static_features)
```

## Variable Mapping Implementation

When implementing a new dataset, map StandardVariable constants to dataset-specific names:

```python
_dynamic_variable_mapping = {
    StandardVariable.STREAMFLOW: {
        "default_source": "observations",
        "sources": {
            "observations": {"specific_name": "q_cms", "unit": "m^3/s"},
            "simulated": {"specific_name": "q_sim", "unit": "m^3/s"},
        },
    },
    StandardVariable.PRECIPITATION: {
        "default_source": "gauge",
        "sources": {
            "gauge": {"specific_name": "precip_mm", "unit": "mm/day"},
            "era5": {"specific_name": "tp_era5", "unit": "mm/day"},
            "chirps": {"specific_name": "precip_chirps", "unit": "mm/day"},
        },
    },
}
```

## API Reference

::: hydrodataset.hydro_dataset.StandardVariable
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
