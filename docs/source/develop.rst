Development
============================
This is a tutorial for developer.

Design strategy
--------------------
The whole structure:

* docs: the document for HydroBench
* hydrobench: the directory of core code
* test: all testing code and data

Let's dive into the "hydrobench" directory.

We could set three individual parts for three datasets respectively.
However, these parts share many common code, hence, we think it's a better way to design the structure
according to the concrete data processing at first. We'll see if it is still good when nearly finishing.

Now the directories in "hydrobench" are:

* app: scripts for downloading and preprocessing source data
* data: this directory contains the data-source classes for source data;
* daymet4basins: this is the directory for downloading and processing daymet data for camels-basins' boundaries;
* ecmwf4basins: this is the directory for downloading and processing ERA5 data;
* modis4basins: this is the directory for downloading and processing MODIS products data;
* nldas4basins: this is the directory for downloading and processing NLDAS data;
* pet: all methods for calculating potential evapotranspiration are here;
* utils: all utility functions.

Conventions
----------------------
* Please write unittest code for all functions.
* Please write comments by English, and note we use the the same format in `this example <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy>`_.

Update documents
----------------------
After adding new features or modifying existed code, please update the document:

.. code-block:: shell

    cd docs
    sphinx-apidoc -o source/ ../hydrobench
    make html

