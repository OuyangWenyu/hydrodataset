# HydroBench

A benchmark dataset for data-driven hydrologic forecasting

This repository contains all the code for downloding and processing the data as well as code for the baseline models.

I draw lessons from [WeatherBench](https://github.com/pangeo-data/WeatherBench).

## Leaderborad

| Model | NSE | Notes | Reference | Code |
|--------------------|----------------------------------|----------------------------|------------------|------------------|
| CudnnLSTM | 0.74 | an LSTM model with a good dropout strategy | [Ouyang et al. 2021](https://arxiv.org/abs/2101.04423) |[HydroSPDB](https://github.com/OuyangWenyu/HydroSPDB)|

## Quick start

TODO: write a notebook to quickly start

## Download the data

TODO: put all the dataset to public website

## Baselines and evaluation

IMPORTANT: The format of the predictions file is a NetCDF dataset with dimensions [init_time, lead_time, lat, lon]. Consult the notebooks for examples. You are stongly encouraged to format your predictions in the same way and then use the same evaluation functions to ensure consistent evaluation.

### Baselines

TODO: The baselines are created using Jupyter notebooks in notebooks/. In all notebooks, the forecasts are saved as CSV files in the predictions directory of the dataset.

### LSTM baselines

TODO: An example of how to load the data and train an LSTM 

### Evaluation

TODO: Evaluation and comparison of the different baselines in done

## Data processing

The dataset already contains the most important processed data. If you would like to download a different variable , regrid to a different resolution or extract single levels from the 3D files, here is how to do that!

### Downloading the CAMELS baseline

TODO: CAMELS baselines

### Downloading and processing the raw data from the Daymet dataset

The workflow to get to the processed data that ended up in the data repository above is:

- Download daily files from the Daymet archive (src/download.py)
- Regrid the raw data to the required resolutions (src/regrid.py)
- The raw data is from the ERA5 reanalysis archive. 
  
Information on how to download the data can be found here and here.
