# Download MODIS data

## Method 1

Please refer to this tutorial for general MODIS data downloading:
https://git.earthdata.nasa.gov/projects/LPDUR/repos/daac_data_download_python/browse

If you can read Chinese, please see this tutorial:
https://github.com/OuyangWenyu/aqualord/blob/master/MODIS/1-download-modis.md

Then, put the downloading list file in this directory, open the terminal in this directory and activate the conda env:

```shell
conda activate HydroBench
```

Next, add a .netrc file in your home directory (for example, mine is C:\Users\11445) like this:

```netrc
machine urs.earthdata.nasa.gov
login <your username>
password <your password>
```

Change the username and password to yours.

Perform the following code:

```Shell
python DAACDataDownload.py -dir <insert local directory to save files to> -f <insert a single granule URL, or the location of a csv or text file containing granule URLs>
```

For example:

```Shell
python G:\\Code\\HydroBench\\hydrobench\\modis4basins\\DAACDataDownload.py -dir F:\\data\\mcd15a3hv006 -f G:\\Code\\HydroBench\\hydrobench\\modis4basins\\6063460452-download.txt
```
## Method 2

Use ge_modis.py.

You also need to create .netrc file in the home directory.

Then open ternimal in this directory and perform code like this:

```shell
get_modis.py -v -p MOD13Q1.006 -s MOLT -y 2020 -t h27v06 -o F:\\data\\MOD13Q1 -b 1 -e 180
```

see more [here](https://jgomezdans.github.io/downloading-modis-data-with-python.html)
