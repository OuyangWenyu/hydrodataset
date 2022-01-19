import collections
import os
import time

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import transform, CRS, Proj
from shapely.geometry import Polygon, Point
import xarray as xr

from hydrodataset.utils.hydro_utils import serialize_geopandas


def split_shp_to_shps_in_time_zones(
    basins_shp_file, gages_in_time_zones_dict, save_dir
):
    """
    Split basins' shapefile to multiple shapefiles in each zone

    Parameters
    ----------
    basins_shp_file
        the shape file of all basins;
        Now we only support shapefiles of CAMELS
    gages_in_time_zones_dict
        A dict showing which gages in each time zone;
        got from the "gage_intersect_time_zone" function in this module
    save_dir
        where we save the shapefiles

    Returns
    -------
    None
    """
    basins = gpd.read_file(basins_shp_file)
    basins_ids = [str(station_id).zfill(8) for station_id in basins["hru_id"].values]
    for key in gages_in_time_zones_dict:
        zone_ids = gages_in_time_zones_dict[key]
        zone_basin_ids = np.intersect1d(zone_ids, basins_ids)
        if zone_basin_ids.size < 1:
            continue
        else:
            zone_basin_ids_int = zone_basin_ids.astype(int)
            zone_basin_gdf = basins[basins["hru_id"].isin(zone_basin_ids_int)]
            serialize_geopandas(
                zone_basin_gdf, os.path.join(save_dir, "Camels_" + key + ".shp")
            )


def gage_intersect_time_zone(gage_shp_file, tz_shp_file) -> dict:
    """
    Find which gages in each time zone;
    Now we only support gages in GAGES-II dataset

    Parameters
    ----------
    gage_shp_file
        shapefile of gages
    tz_shp_file
        shapefile of time zones

    Returns
    -------
    dict
        key is time_zone and value are ids of gages in the time zone
    """
    join = spatial_join(gage_shp_file, tz_shp_file)
    zones_df = join["Zone"]
    zones = np.unique(zones_df.values)
    zone_dict = collections.OrderedDict({})
    for zone in zones:
        zone_ids = np.unique(join["STAID"][zones_df[zones_df == zone].index].values)
        zone_dict = collections.OrderedDict({**zone_dict, **{zone: zone_ids.tolist()}})
    return zone_dict


def spatial_join(points_file, polygons_file):
    """join polygons layer to point layer, add polygon which the point is in to the point"""

    points = gpd.read_file(points_file)
    polys = gpd.read_file(polygons_file)
    # Check the data
    if not (points.crs == polys.crs):
        points = points.to_crs(polys.crs)

    # Make a spatial join
    join = gpd.sjoin(points, polys, how="inner", op="within")
    return join


def crd2grid(y, x):
    ux, indX0, indX = np.unique(x, return_index=True, return_inverse=True)
    uy, indY0, indY = np.unique(y, return_index=True, return_inverse=True)

    minDx = np.min(ux[1:] - ux[0:-1])
    minDy = np.min(uy[1:] - uy[0:-1])
    maxDx = np.max(ux[1:] - ux[0:-1])
    maxDy = np.max(uy[1:] - uy[0:-1])
    if maxDx > minDx * 2:
        print("skipped rows")
    #     indMissX=np.where((ux[1:]-ux[0:-1])>minDx*2)[0]
    #     insertX=(ux[indMissX+1]+ux[indMissX])/2
    #     ux=np.insert(ux,indMissX,insertX)
    if maxDy > minDy * 2:
        print("skipped coloums")
    #     indMissY=np.where((uy[1:]-uy[0:-1])>minDy*2)
    #     raise Exception('skipped coloums or rows')

    uy = uy[::-1]
    ny = len(uy)
    indY = ny - 1 - indY
    return (uy, ux, indY, indX)


def array2grid(data, *, lat, lon):
    (uy, ux, indY, indX) = crd2grid(lat, lon)
    ny = len(uy)
    nx = len(ux)
    if data.ndim == 2:
        nt = data.shape[1]
        grid = np.full([ny, nx, nt], np.nan)
        grid[indY, indX, :] = data
    elif data.ndim == 1:
        grid = np.full([ny, nx], np.nan)
        grid[indY, indX] = data
    return grid, uy, ux


def trans_points(from_crs, to_crs, pxs, pys):
    """put the data into dataframe so that the speed of processing could be improved obviously
    :param
    pxs: x of every point (list/array)
    pys: y of every point (list/array)
    :return
    pxys_out: x and y compared a pair list to initialize a polygon
    """
    df = pd.DataFrame({"x": pxs, "y": pys})
    start = time.time()
    df["x2"], df["y2"] = transform(from_crs, to_crs, df["x"].tolist(), df["y"].tolist())
    end = time.time()
    print("time consuming：", "%.7f" % (end - start))
    # after transforming xs and ys, pick out x2, y2，and tranform to numpy array，then do a transportation. Finally put coordination of every row to a list
    arr_x = df["x2"].values
    arr_y = df["y2"].values
    pxys_out = np.stack((arr_x, arr_y), 0).T
    return pxys_out


def trans_polygon(from_crs, to_crs, polygon_from):
    """transform coordination of every point of a polygon to one in a given coordination system"""
    polygon_to = Polygon()
    # data type: tuples in a list
    boundary = polygon_from.boundary
    boundary_type = boundary.geom_type
    print(boundary_type)
    if boundary_type == "LineString":
        pxs = polygon_from.exterior.xy[0]
        pys = polygon_from.exterior.xy[1]
        pxys_out = trans_points(from_crs, to_crs, pxs, pys)
        polygon_to = Polygon(pxys_out)
    elif boundary_type == "MultiLineString":
        # if there is interior boundary in a polygon，then we need to transform its coordinations. Notice: maybe multiple interior boundaries exist.
        exts_x = boundary[0].xy[0]
        exts_y = boundary[0].xy[1]
        pxys_ext = trans_points(from_crs, to_crs, exts_x, exts_y)

        pxys_ints = []
        for i in range(1, len(boundary)):
            ints_x = boundary[i].xy[0]
            ints_y = boundary[i].xy[1]
            pxys_int = trans_points(from_crs, to_crs, ints_x, ints_y)
            pxys_ints.append(pxys_int)

        polygon_to = Polygon(shell=pxys_ext, holes=pxys_ints)
    else:
        print("error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    return polygon_to


def write_shpfile(geodata, output_folder, id_str="hru_id"):
    """generate a shpfile from geodataframe，the name is id of the pandas dataframe"""
    # Create a output path for the data
    gage_id = geodata.iloc[0, :][id_str]
    # id is number，here turn it to str
    output_file = str(int(gage_id)).zfill(8)
    output_fp = os.path.join(output_folder, output_file + ".shp")
    # Write those rows into a new file (the default output file format is Shapefile)
    geodata.to_file(output_fp)


def trans_shp_coord(
    input_folder, input_shp_file, output_folder, output_crs_epsg_or_proj4_str="4326"
):
    """tranform a shapefile to a target coord，default target coord is WGS84:  +proj=longlat +datum=WGS84 +no_defs"""
    # Join folder path and filename
    fp = os.path.join(input_folder, input_shp_file)
    data = gpd.read_file(fp)
    # crs_proj4 = CRS(data.crs).to_proj4()
    crs_proj4 = CRS(data.crs)
    # crs_final = CRS.from_proj4(output_crs_proj4_str)
    # Proj must be used，if not, maybe x represent longitude, and the other represent latitude and it's wrong
    crs_final = Proj(init="epsg:" + output_crs_epsg_or_proj4_str)
    # crs_final = CRS.from_epsg(output_crs_epsg_or_proj4_str)
    all_columns = data.columns.values  # ndarray type
    new_datas = []
    start = time.time()
    for i in range(0, data.shape[0]):  # data.shape[0]
        print("the  ", i, "st basin's shapefile:")
        newdata = gpd.GeoDataFrame()
        for column in all_columns:
            # when read shapefile using geodataframe, the name of geo column is "geometry"
            if column == "geometry":
                # first change the coord
                polygon_from = data.iloc[i, :]["geometry"]
                polygon_to = trans_polygon(crs_proj4, crs_final, polygon_from)
                # assign value to location i of newdata，if not it will be geoseries，which cannot be imported to shapefile
                newdata.at[0, "geometry"] = polygon_to
                print(type(newdata.at[0, "geometry"]))
            else:
                newdata.at[0, column] = data.iloc[i, :][column]
        print("coordination transform！")
        print(newdata)
        # must use fiona's crs to guarantee the result is correct
        newdata.crs = fiona.crs.from_epsg(int(output_crs_epsg_or_proj4_str))
        print("Coordination system: ", newdata.crs)
        new_datas.append(newdata)
        write_shpfile(newdata, output_folder)
    end = time.time()
    print("time consuming：", "%.7f" % (end - start))
    return new_datas


def nearest_point_index(crs_from, crs_to, lon, lat, xs, ys):
    # x and y are proj coord，lon, lat should be transformed (x is longtitude projection，y is lat)
    x, y = transform(crs_from, crs_to, lon, lat)
    index_x = (np.abs(xs - x)).argmin()
    index_y = (np.abs(ys - y)).argmin()
    return [index_x, index_y]


def create_mask(poly, xs, ys, lons, lats, crs_from, crs_to):
    mask_index = []
    poly_bound = poly.bounds
    poly_bound_min_lat = poly_bound[1]
    poly_bound_min_lon = poly_bound[0]
    poly_bound_max_lat = poly_bound[3]
    poly_bound_max_lon = poly_bound[2]
    index_min = nearest_point_index(
        crs_from, crs_to, poly_bound_min_lon, poly_bound_min_lat, xs, ys
    )
    index_max = nearest_point_index(
        crs_from, crs_to, poly_bound_max_lon, poly_bound_max_lat, xs, ys
    )
    range_x = [index_min[0], index_max[0]]
    range_y = [index_max[1], index_min[1]]
    for i in range(range_y[0], range_y[1] + 1):
        for j in range(range_x[0], range_x[1] + 1):
            if is_point_in_boundary(lons[i][j], lats[i][j], poly):
                mask_index.append((i, j))
    return mask_index


def is_point_in_boundary(px, py, poly):
    point = Point(px, py)
    return point.within(poly)


def shps_trans_coord(input_folder, output_folder):
    """transform coords of all shapefiles in the folder--"input_folder",
    and save the results in the folder--"output_folder"
    """
    # Define path to folder
    shp_file_names = []
    for f_name in os.listdir(input_folder):
        if f_name.endswith(".shp"):
            shp_file_names.append(f_name)

    for i in range(len(shp_file_names)):
        shp_file = shp_file_names[i]
        # output_folder = r"examples_data/wgs84lccsp2" crs_final_str = '+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5
        # +lon_0=-100 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
        new_datas = trans_shp_coord(input_folder, shp_file, output_folder)


def basin_avg_netcdf(netcdf_file, shp_file, mask_file):
    # TODO: use xarray and dask
    data_netcdf = xr.open_dataset(netcdf_file)  # reads the netCDF file
    temp_lat = data_netcdf.variables["lat"]  # temperature variable
    temp_lon = data_netcdf.variables["lon"]  # temperature variable
    for d in data_netcdf.dimensions.items():
        print(d)
    x, y = data_netcdf.variables["x"], data_netcdf.variables["y"]
    x = data_netcdf.variables["x"][:]
    y = data_netcdf.variables["y"][:]
    lx = list(x)
    ly = list(y)
    print(all(ix < jx for ix, jx in zip(lx, lx[1:])))
    print(all(iy > jy for iy, jy in zip(ly, ly[1:])))
    lons = data_netcdf.variables["lon"][:]
    lats = data_netcdf.variables["lat"][:]

    crs_pro_str = "+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    crs_geo_str = "+proj=longlat +datum=WGS84 +no_defs"
    crs_from = CRS.from_proj4(crs_geo_str)
    crs_to = CRS.from_proj4(crs_pro_str)

    new_shps = gpd.read_file(shp_file)
    polygon = new_shps.at[0, "geometry"]
    start = time.time()
    mask = create_mask(polygon, x, y, lons, lats, crs_from, crs_to)
    end = time.time()
    print("time：", "%.7f" % (end - start))
    serialize_numpy(np.array(mask), mask_file)
    var_types = ["tmax"]
    # var_types = ['tmax', 'tmin', 'prcp', 'srad', 'vp', 'swe', 'dayl']
    avgs = []
    for var_type in var_types:
        start = time.time()
        avg = calc_avg(mask, data_netcdf, var_type)
        end = time.time()
        print("time：", "%.7f" % (end - start))
        print("mean value：", avg)
        avgs.append(avg)

    return avgs


def ind_of_dispersion(coord, points):
    """the ratio of variance and mean value of Euclidean distances between event points and a selected point"""
    points = np.asarray(points)
    xd = points[:, 0] - coord[0]
    yd = points[:, 1] - coord[1]
    mean_d = np.sqrt(xd * xd + yd * yd).mean()
    var_d = np.sqrt(xd * xd + yd * yd).var()
    ind = var_d / mean_d
    return ind


def coefficient_of_variation(coord, points):
    """the ratio of the standard deviation to the mean (average) value of Euclidean distances between event points
    and a selected point"""
    if len(points) == 0:
        return np.nan
    points = np.asarray(points)
    xd = points[:, 0] - coord[0]
    yd = points[:, 1] - coord[1]
    mean_d = np.sqrt(xd * xd + yd * yd).mean()
    var_d = np.sqrt(xd * xd + yd * yd).var()
    coefficient = np.sqrt(var_d) / mean_d * 100
    return coefficient
