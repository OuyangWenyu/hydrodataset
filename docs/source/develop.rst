Development
============================
This is a tutorial for developer.

Design strategy
--------------------
The whole structure:

* app: scripts for downloading and preprocessing source data
* docs: the document for HydroBench
* example: some sample data
* hydrobench: the directory of core code
* test: all testing code using Unit test

Let's dive into the "hydrobench" directory.

We could set three individual parts for three datasets respectively.
However, these parts share many common code, hence, we think it's a better way to design the structure
according to the concrete data processing at first. We'll see if it is still good when nearly finishing.

Now the directories in "hydrobench" are:

* data: this directory contains the data-source classes for source data;
* daymet4basins: this is the directory for downloading daymet data for basins' boundaries;
* pet: all methods for calculating potential evapotranspiration are here;
* utils: all utility functions.

Conventions
----------------------
* Please write unittest code for all functions.
* Please write comments by English, and note we use the the same format in `this example <https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html#an-example-class-with-docstrings>`_.
