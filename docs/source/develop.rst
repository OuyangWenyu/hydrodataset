Development
============================
This is a tutorial for developer.

Design strategy
--------------------
The whole structure:

* docs: the document for HydroDatasets
* hydrodatasets: the directory of core code
* test: all testing code and data

Now the directories in "hydrodatasets" include:

* app: scripts for downloading and preprocessing source data
* data: this directory contains the data-source classes for source data;
* daymet4basins: this is the directory for downloading and processing daymet data for camels-basins' boundaries;
* ecmwf4basins: this is the directory for downloading and processing ERA5-Land data;
* climateproj4basins: this is the directory for downloading and processing NEX-DCP30 data;
* modis4basins: this is the directory for downloading and processing MODIS ET products data;
* nldas4basins: this is the directory for downloading and processing NLDAS forcing data;
* pet: all methods for calculating potential evapotranspiration are here;
* utils: all utility functions.

Conventions
----------------------
* Please write pytest code for all your defined functions.
* Please write comments by English, and note we use the the same format in `this example <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy>`_.

Update documents
----------------------
After adding new features or modifying existed code, please update the document:

.. code-block:: shell

    cd docs
    sphinx-apidoc -o source/ ../hydrodatasets
    make html

