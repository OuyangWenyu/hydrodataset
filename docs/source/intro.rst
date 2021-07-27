Introduction
============================
This is a dataset zoo for hydrological modeling, especially with data-driven methods.

Overview
--------------------
Now we are making three datasets. After finishing the preparation of datasets, we will provide deep-learning benchmark experiments for each dataset.

 * GHS: Geospatial attributes and Hydrometeorological forcings of gages for Streamflow modeling
 * CAMELS-grid: a grid-version CAMELS dataset
 * HFlowDB: a DataBase for Hourly streamflow modeling of gages, especially for High flow (Flood) modeling.

The "GHS" is an extension for `the CAMELS dataset <https://ral.ucar.edu/solutions/products/camels>`_ .
It contains geospatial attributes, hydrometeorological forcings and streamflow data of 9067 gages over the Contiguous United States (CONUS)
in `the GAGES-II dataset <https://water.usgs.gov/GIS/metadata/usgswrd/XML/gagesII_Sept2011.xml>`_ .

CAMELS-grid include all basins in the CAMELS dataset. We create boundary masks for each basin
and download the gridded data of `the Daymet dataset <https://daymet.ornl.gov/>`_ for all of them.

HFlowDB is mainly created for flood forecasting in basins. It origins from `the FlowDB dataset <https://arxiv.org/abs/2012.11154>`_ .
But HFlowDB is for gages and its basins, while FlowDB only include streamflow data of gages and precipitation data of its nearest rainfall gage.

Download
-------------------
Now we have not provided an online way to download the data.
If you need the dataset, please contact with `Ouyang, Wenyu <https://github.com/OuyangWenyu>`_

Contributor
-------------------
Wenyu Ouyang: hust2014owen@gmail.com
Tianqi Xin: 2034687001@qq.com
