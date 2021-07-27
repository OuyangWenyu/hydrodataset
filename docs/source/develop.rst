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

 * data:
 * daymet4basins:
 * pet:
 * utils:

Conventions
----------------------
Please write unittest code for all functions.
Please write comments by English, and note we use the the format.

