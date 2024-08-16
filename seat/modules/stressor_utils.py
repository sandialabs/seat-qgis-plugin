"""
/***************************************************************************.

 velocity_module.py
 Copyright 2023, Integral Consulting Inc. All rights reserved.

 PURPOSE: definitions used by the stressor modules

 PROJECT INFORMATION:
 Name: SEAT - Spatial and Environmental Assessment Toolkit
 Number: C1308

 AUTHORS
  Timothy Nelson (tnelson@integral-corp.com)
  Sam McWilliams (smcwilliams@integral-corp.com)
  Eben Pendelton

 NOTES (Data descriptions and any script specific notes)
	1. called by shear_stress_module.py, velocity_module.py, acoustics_module.py
"""

import numpy as np
import pandas as pd
from matplotlib.tri import LinearTriInterpolator, TriAnalyzer, Triangulation
from scipy.interpolate import griddata
from osgeo import gdal, osr
import os


def estimate_grid_spacing(x, y, nsamples=100):
    """
    Estimates the grid spacing of an unstructured grid to create a similar resolution structured grid

    Parameters
    ----------
    x : array
        x-coordinates.
    y : array
        y-coordiantes.
    nsamples : scalar, optional
        number of random points to sample spacing to nearest cell. The default is 100.

    Returns
    -------
    dxdy : array
        estimated median spacing between samples.

    """
    import random
    import sys

    random.seed(10)
    coords = list(set(zip(x, y)))
    if nsamples != len(x):
        points = [random.choice(coords)
                  for i in range(nsamples)]  # pick N random points
    else:
        points = coords
    MD = []
    for p0x, p0y in points:
        minimum_distance = sys.maxsize
        for px, py in coords:
            distance = np.sqrt((p0x - px) ** 2 + (p0y - py) ** 2)
            if (distance < minimum_distance) & (distance != 0):
                minimum_distance = distance
        MD.append(minimum_distance)
    dxdy = np.median(MD)
    return dxdy


def create_structured_array_from_unstructured(x, y, z, dxdy, flatness=0.2):
    """
    Creates a structured grid from an unsructred grid

    Parameters
    ----------
    x : array
        input x-coordinates.
    y : array
        input y-coordiantes.
    z : array
        input value.
    dxdy : scalar
        spacing between x and y.
    flatness : scalar, optional
        DESCRIPTION. The default is 0.2.

    Returns
    -------
    refxg : array
        x-coordinate.
    refyg : array
        y-coordiante.
    z : array
        interpolated z value.

    """
    # flatness is from 0-.5 .5 is equilateral triangle
    refx = np.arange(np.nanmin(x), np.nanmax(x)+dxdy, dxdy)
    refy = np.arange(np.nanmin(y), np.nanmax(y)+dxdy, dxdy)
    refxg, refyg = np.meshgrid(refx, refy)
    tri = Triangulation(x, y)
    mask = TriAnalyzer(tri).get_flat_tri_mask(flatness)
    tri.set_mask(mask)
    tli = LinearTriInterpolator(tri, z)
    z_interp = tli(refxg, refyg)
    return refxg, refyg, z_interp.data


def redefine_structured_grid(x, y, z):
    """
    ensures regular structured grid (paracousti grid not regular)

    Parameters
    ----------
    x : array
        x-coordinates.
    y : array
        y-coordiantes.
    z : array
        input value.

    Returns
    -------
    x_new : array
       x-coordinate.
    y_new : array
       y-coordinate.
    z_new : array
        interpolated z value..

    """
    min_x = np.nanmin(x)
    min_y = np.nanmin(y)
    max_x = np.nanmax(x)
    max_y = np.nanmax(y)
    xx = np.linspace(min_x, max_x, x.shape[1])
    yy = np.linspace(min_y, max_y, y.shape[0])
    dx = np.nanmin([np.nanmedian(np.diff(xx)), np.nanmedian(np.diff(yy))])
    xx = np.arange(min_x, max_x+dx, dx)
    yy = np.arange(min_y, max_y+dx, dx)
    x_new, y_new = np.meshgrid(xx, yy)
    z_new = griddata((x.flatten(), y.flatten()), z.flatten(),
                     (x_new, y_new), method='nearest', fill_value=0)
    return x_new, y_new, z_new


def resample_structured_grid(x_grid, y_grid, z, X_grid_out, Y_grid_out, interpmethod='nearest'):
    """
    interpolates a structured grid onto a new structured grid.

    Parameters
    ----------
    x_grid : array
        x-coordinates.
    y_grid : array
        y-coordinates.
    z : array
        input value.
    X_grid_out : array
        x-coordinates.
    Y_grid_out : array
        y-coordinates.
    interpmethod : str, optional
        interpolation method to use. The default is 'nearest'.

    Returns
    -------
    array
        interpolated z-value on X/Y _grid_out.

    """
    return griddata((x_grid.flatten(), y_grid.flatten()), z.flatten(), (X_grid_out, Y_grid_out), method=interpmethod, fill_value=0)


def calc_receptor_array(receptor_filename, x, y, latlon=False, mask=None):
    """
    Creates an array from either a .tif or .csv file.

    Parameters
    ----------
    receptor_filename : str
        File path to the recetptor file (*.csv or *.tif).
    x : array
        x-coordinates to interpolate at.
    y : array
        y-coordiantes to interpolate at.
    latlon :  Bool, optional
        True is coordinates are lat/lon. The default is False.

    Raises
    ------
    Exception
        "Invalid Recetpor File Type. Must be of type .tif or .csv".

    Returns
    -------
    receptor_array : array
        interpolated receptor values at input x/y coordinates.

    """
    # if ((receptor_filename is not None) or (not receptor_filename == "")):
    if not ((receptor_filename is None) or (receptor_filename == "")):
        if receptor_filename.endswith('.tif'):
            data = gdal.Open(receptor_filename)
            img = data.GetRasterBand(1)
            receptor_array = img.ReadAsArray()
            receptor_array[receptor_array < 0] = 0
            (upper_left_x, x_size, x_rotation, upper_left_y,
             y_rotation, y_size) = data.GetGeoTransform()
            cols = data.RasterXSize
            rows = data.RasterYSize
            r_rows = np.arange(rows) * y_size + upper_left_y + (y_size / 2)
            r_cols = np.arange(cols) * x_size + upper_left_x + (x_size / 2)
            if latlon == True:
                r_cols = np.where(r_cols < 0, r_cols+360, r_cols)
            x_grid, y_grid = np.meshgrid(r_cols, r_rows)
            receptor_array = griddata((x_grid.flatten(), y_grid.flatten(
            )), receptor_array.flatten(), (x, y), method='nearest', fill_value=0)

        elif receptor_filename.endswith('.csv'):
            receptor_array = pd.read_csv(
                receptor_filename, header=None, index_col=0).to_numpy().item() * np.ones(x.shape)
        else:
            raise Exception(
                f"Invalid Receptor File {receptor_filename}. Must be of type .tif or .csv")
    else:
        # taucrit without a receptor
        # Assume the following grain sizes and conditions for typical beach sand (Nielsen, 1992 p.108)
        receptor_array = 200*1e-6 * np.ones(x.shape)
    if mask is not None:
        receptor_array = np.where(mask, receptor_array, np.nan)
    return receptor_array


def trim_zeros(x, y, z1, z2):
    """
    removes zeros from velocity structure array [might not always occur but does for test data]

    Parameters
    ----------
    x : array
        x-coordinates.
    y : array
        y-coordiantes.
    z1 : array
        first value (e.g. uvar).
    z2 : array
        second value (e.g. vvar).

    Returns
    -------
    array
        x-coordinate.
    array
        y-coordinate.
    array
        z1-value (e.g. uvar).
    array
        z2-value (e.g. vvar).

    """
    # edges of structured array have zeros, not sure if universal issue
    return x[1:-1, 1:-1], y[1:-1, 1:-1], z1[:, :, 1:-1, 1:-1], z2[:, :, 1:-1, 1:-1]


def create_raster(
    output_path,
    cols,
    rows,
    nbands,
    eType=gdal.GDT_Float32,
):
    """
    Create a gdal raster object.

    Parameters
    ----------
    output_path :  str
        Absolute filepath of geotiff to create.
    cols : scalar
        number of columns.
    rows : scalar
        number of rows.
    nbands : scalar
        number of bads to write.
    eType : gdal, optional
        type of geotiff and precision. The default is gdal.GDT_Float32.

    Returns
    -------
    output_raster : gdal raster object
        gdal raster object.

    """

    # create gdal driver - doing this explicitly
    driver = gdal.GetDriverByName(str("GTiff"))

    output_raster = driver.Create(
        output_path,
        int(cols),
        int(rows),
        nbands,
        eType=gdal.GDT_Float32,
    )

    # spatial_reference = osr.SpatialReference()
    # spatial_reference.ImportFromEPSG(spatial_reference_system_wkid)
    # output_raster.SetProjection(spatial_reference.ExportToWkt())

    # returns gdal data source raster object
    return output_raster


def numpy_array_to_raster(
    output_raster,
    numpy_array,
    bounds,
    cell_resolution,
    spatial_reference_system_wkid,
    output_path,
    nodata_val=None
):
    """


    Parameters
    ----------
    output_raster : gdal raster object
        gdal raster object to write array.
    numpy_array : array
        numpy array to save to geotiff.
    bounds : array
        [xmin, ymin].
    cell_resolution : array
        [dx, dy].
    spatial_reference_system_wkid : scalar
        EPSG code.
    output_path : str
        Absolute filepath of geotiff to create.

    Raises
    ------
    Exception
        "Failed to create raster: %s" % output_path.

    Returns
    -------
    output_path : str
         filepath of geotiff created.

    """

    """Create the output raster."""
    # create output raster
    # (upper_left_x, x_resolution, x_skew 0, upper_left_y, y_skew 0, y_resolution).
    geotransform = (
        bounds[0],
        cell_resolution[0],
        0,
        bounds[1] + cell_resolution[1],
        0,
        -1 * cell_resolution[1],
    )

    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(spatial_reference_system_wkid)

    output_raster.SetProjection(
        spatial_reference.ExportToWkt(),
    )  # exports the cords to the file
    output_raster.SetGeoTransform(geotransform)
    output_band = output_raster.GetRasterBand(1)
    if nodata_val is not None:
        output_band.SetNoDataValue(nodata_val) #Not an issue, may be in other cases?
    output_band.WriteArray(numpy_array)

    output_band.FlushCache()
    output_band.ComputeStatistics(
        False,
    )  # you want this false, true will make computed results, but is faster, could be a setting in the UI perhaps, esp for large rasters?

    if os.path.exists(output_path) == False:
        raise Exception("Failed to create raster: %s" % output_path)

    # this closes the file
    output_raster = None
    return output_path


def find_utm_srid(lon, lat, srid):
    """
    Given a WGS 64 srid calculate the corresponding UTM srid.

    Parameters
    ----------
    lon : array
        WGS 84 (srid 4326) longitude value
    lat : array
        WGS 84 (srid 4326) latitude value.
    srid : scalar
        WGS 84 srid 4326 to make sure the function is appropriate.

    Returns
    -------
    None.

    """

    assert srid == 4326, f"find_utm_srid: input geometry has wrong SRID {srid}"

    if lat < 0:
        # south hemisphere
        base_srid = 32700
    else:
        # north hemisphere or on equator
        base_srid = 32600

    # calculate final srid
    out_srid = base_srid + np.floor((lon + 186) / 6)

    if lon == 180:
        out_srid = base_srid + 60

    return out_srid


def read_raster(raster_name):
    """
    Reads a raster to an array.

    Parameters
    ----------
    raster_name : str
        absolute filepath of array.

    Returns
    -------
    rx : array
        x-coordinates.
    ry : array
        y-coordinate.
    raster_array : array
        value array.

    """

    data = gdal.Open(raster_name)
    img = data.GetRasterBand(1)
    raster_array = img.ReadAsArray()
    (upper_left_x, x_size, x_rotation, upper_left_y,
     y_rotation, y_size) = data.GetGeoTransform()
    cols = data.RasterXSize
    rows = data.RasterYSize
    r_rows = np.arange(rows) * y_size + upper_left_y + (y_size / 2)
    r_cols = np.arange(cols) * x_size + upper_left_x + (x_size / 2)
    rx, ry = np.meshgrid(r_cols, r_rows)
    data = None
    return rx, ry, raster_array

def secondary_constraint_geotiff_to_numpy(filename):
    data = gdal.Open(filename)
    img = data.GetRasterBand(1)
    array = img.ReadAsArray()

    (upper_left_x, x_size, x_rotation, upper_left_y,
        y_rotation, y_size) = data.GetGeoTransform()
    cols = data.RasterXSize
    rows = data.RasterYSize
    r_rows = np.arange(rows) * y_size + upper_left_y + (y_size / 2)
    r_cols = np.arange(cols) * x_size + upper_left_x + (x_size / 2)
    prj = data.GetProjection()
    srs = osr.SpatialReference(wkt=prj)
    if srs.GetAttrValue('AUTHORITY',1) == '4326':
        r_cols = np.where(r_cols < 0, r_cols+360, r_cols)
    x_grid, y_grid = np.meshgrid(r_cols, r_rows)
    return x_grid, y_grid, array

def calculate_cell_area(rx, ry, latlon=True):
    """
    Calculates the area of each cell

    Parameters
    ----------
    rx : array
        x-coordinate array.
    ry : array
        y-coordinate array.
    latlon : Bool, optional
        True is coordinates are lat/lon. The default is True.

    Returns
    -------
    rxm : TYPE
        x-coordinate array at cell center.
    rym : TYPE
        y-coordinate array at cell center.
    square_area : array
        area of each cell.

    """
    import numpy as np
    from pyproj import Geod

    if latlon == True:
        geod = Geod(ellps="WGS84")
        lon2D, lat2D = np.where(rx > 180, rx-360, rx), ry
        _, _, distEW = geod.inv(
            lon2D[:, :-1], lat2D[:, 1:], lon2D[:, 1:], lat2D[:, 1:])
        _, _, distNS = geod.inv(
            lon2D[1:, :], lat2D[1:, :], lon2D[1:, :], lat2D[:-1, :])
        square_area = distEW[1:, :] * distNS[:, 1:]
    else:
        square_area = np.zeros((rx.shape[0]-1, ry.shape[1]-1))
        for row in range(rx.shape[0]-1):
            for col in range(ry.shape[1]-1):
                dx = rx[row, col+1] - rx[row, col]
                dy = ry[row+1, col] - ry[row, col]
                square_area[row, col] = dx*dy
    rxm = np.zeros(square_area.shape)
    rym = np.zeros(square_area.shape)
    for row in range(rx.shape[0]-1):
        rxm[row, :] = (rx[row, :-1] + rx[row, 1:])/2
    for col in range(ry.shape[1]-1):
        rym[:, col] = (ry[:-1, col] + ry[1:, col])/2

    return rxm, rym, square_area


def bin_data(zm, square_area, nbins=25):
    """
    Bin statistics and area calculation of binned values.

    Parameters
    ----------
    zm : array
        value array (ensure same dimension as square_area).
    square_area : array
        square area array (output of calculate_cell_area).
    nbins : scalar, optional
        number of bins to calculate. The default is 25.

    Returns
    -------
    DATA : Dictionary
        Dictionary for each bin contating
            bin start : the starting value of each bin
            bin end : the last value of each bin
            bin center: the center value of each bin
            count :number of cells in each bin
            Area : area overwhich the binned values occur

    """
    hist, bins = np.histogram(zm, bins=nbins)
    center = (bins[:-1] + bins[1:]) / 2
    DATA = {}
    DATA['bin start'] = bins[:-1]
    DATA['bin end'] = bins[1:]
    DATA['bin center'] = center
    DATA['count'] = hist
    DATA['Area'] = np.zeros(hist.shape)
    for ic, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
        if ic < len(hist)-1:
            area_ix = np.flatnonzero((zm >= bin_start) & (zm < bin_end))
        else:
            area_ix = np.flatnonzero((zm >= bin_start) & (zm <= bin_end))
        DATA['Area'][ic] = np.sum(square_area[area_ix])
    return DATA


def bin_receptor(zm, receptor, square_area, nbins=25, receptor_names=None):
    """
    Bins values into 25 bins and by unique values in the receptor.

    Parameters
    ----------
    zm : array
        value array (ensure same dimension as square_area).
    receptor : array
        receptor array values (ensure same dimension as square_area).
    square_area : array
        square area array (output of calculate_cell_area).
    nbins : scalar, optional
        number of bins to calculate. The default is 25.
        receptor_names
    receptor_names : list, opional
        optional names for each unique value in the receptor. the default is None.

    Returns
    -------
    DATA : Dictionary
        Dictionary contating with keyscorresponding to each unique receptor value each containing for each bin
            bin start : the starting value of each bin
            bin end : the last value of each bin
            bin center: the center value of each bin
            count :number of cells in each bin
            Area : area overwhich the binned values occur.

    """
    hist, bins = np.histogram(zm, bins=nbins)
    center = (bins[:-1] + bins[1:]) / 2
    DATA = {}
    DATA['bin start'] = bins[:-1]
    DATA['bin end'] = bins[1:]
    DATA['bin center'] = center
    for ic, rval in enumerate(np.unique(receptor)):
        zz = zm[receptor == rval]
        sqa = square_area[receptor == rval]
        rcolname = f'Area, receptor value {rval}' if receptor_names is None else receptor_names[
            ic]
        DATA[rcolname] = np.zeros(hist.shape)
        for ic, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
            if ic < len(hist)-1:
                area_ix = np.flatnonzero((zz >= bin_start) & (zz < bin_end))
            else:
                area_ix = np.flatnonzero((zz >= bin_start) & (zz <= bin_end))
            DATA[rcolname][ic] = np.sum(sqa[area_ix])
        DATA[f'Area percent, receptor value {rval}'] = 100 * \
            DATA[rcolname]/DATA[rcolname].sum()
    return DATA


def bin_layer(raster_filename, receptor_filename=None, receptor_names=None, limit_receptor_range=None, latlon=True):
    """
    creates a dataframe of binned raster values and associtaed area and percent of array.

    Parameters
    ----------
    raster_filename : str
        file path and name of raster.
    receptor_filename : str, optional
        file path and name of raster. The default is None.
    receptor_names : list, opional
        optional names for each unique value in the receptor. The default is None.
    limit_receptor_range : array, optional
        Range over which to limit uniuqe raster values [start, stop]. The default is None.
    latlon : Bool, optional
        True is coordinates are lat/lon. The default is True.

    Returns
    -------
    DataFrame
        DataFrame with bins as rows and statistics values as columns.
            [bin stats, area, count, percent]

    """
    rx, ry, z = read_raster(raster_filename)
    rxm, rym, square_area = calculate_cell_area(rx, ry, latlon=True)
    square_area = square_area.flatten()
    zm = resample_structured_grid(
        rx, ry, z, rxm, rym, interpmethod='linear').flatten()
    if receptor_filename is None:
        DATA = bin_data(zm[np.invert(np.isnan(zm))],
                        square_area[np.invert(np.isnan(zm))], nbins=25)
        # DF = pd.DataFrame(DATA)
        DATA['Area percent'] = 100 * DATA['Area']/DATA['Area'].sum()
    else:
        rrx, rry, receptor = read_raster(receptor_filename)
        receptor = resample_structured_grid(
            rrx, rry, receptor, rxm, rym).flatten()
        if limit_receptor_range is not None:
            receptor = np.where((receptor >= np.min(limit_receptor_range)) & (
                receptor <= np.max(limit_receptor_range)), receptor, 0)
        DATA = bin_receptor(zm[np.invert(np.isnan(zm))], receptor[np.invert(np.isnan(
            zm))], square_area[np.invert(np.isnan(zm))], receptor_names=receptor_names)
    return pd.DataFrame(DATA)


def classify_layer_area(raster_filename, receptor_filename=None, at_values=None, value_names=None, limit_receptor_range=None, latlon=True):
    """
    Creates a dataframe of raster values and associtaed area and percent of array at specified raster values.

    Parameters
    ----------
    raster_filename : str
        file path and name of raster.
    receptor_filename : str, optional
        file path and name of raster. The default is None.
    at_values : list, optional
        raster values to sample. The default is None.
    value_names : list, optional
        names of unique raster values. The default is None.
    limit_receptor_range : array, optional
        Range over which to limit uniuqe raster values [start, stop]. The default is None.
    latlon : Bool, optional
        True is coordinates are lat/lon. The default is True.


    Returns
    -------
    DataFrame
        DataFrame with sampled values as rows and statistics values as columns.
            [sampled value, stats, area, count, percent]

    """
    rx, ry, z = read_raster(raster_filename)
    rxm, rym, square_area = calculate_cell_area(rx, ry, latlon=latlon)
    square_area = square_area.flatten()
    zm = resample_structured_grid(rx, ry, z, rxm, rym).flatten()
    if at_values is None:
        at_values = np.unique(zm)
    else:
        at_values = np.atleast_1d(at_values)
    DATA = {}
    DATA['value'] = at_values
    if value_names is not None:
        DATA['value name'] = value_names
    if receptor_filename is None:
        DATA['Area'] = np.zeros(len(at_values))
        for ic, value in enumerate(at_values):
            area_ix = np.flatnonzero(zm == value)
            DATA['Area'][ic] = np.sum(square_area[area_ix])
        DATA['Area percent'] = 100 * DATA['Area']/DATA['Area'].sum()
    else:
        rrx, rry, receptor = read_raster(receptor_filename)
        if limit_receptor_range is not None:
            receptor = np.where((receptor >= np.min(limit_receptor_range)) & (
                receptor <= np.max(limit_receptor_range)), receptor, 0)
        receptor = resample_structured_grid(
            rrx, rry, receptor, rxm, rym).flatten()
        for ic, rval in enumerate(np.unique(receptor)):
            zz = zm[receptor == rval]
            sqa = square_area[receptor == rval]
            rcolname = f'Area, receptor value {rval}'
            ccolname = f'Count, receptor value {rval}'
            DATA[rcolname] = np.zeros(len(at_values))
            DATA[ccolname] = np.zeros(len(at_values))
            for iic, value in enumerate(at_values):
                area_ix = np.flatnonzero(zz == value)
                DATA[ccolname][iic] = len(area_ix)
                DATA[rcolname][iic] = np.sum(sqa[area_ix])
            DATA[f'Area percent, receptor value {rval}'] = 100 * \
                DATA[rcolname]/DATA[rcolname].sum()
    return pd.DataFrame(DATA)


def classify_layer_area_2nd_Constraint(raster_to_sample, secondary_constraint_filename, at_raster_values, at_raster_value_names, limit_constraint_range, latlon):
    rx, ry, z = read_raster(raster_to_sample)
    rxm, rym, square_area = calculate_cell_area(rx, ry, latlon=latlon)
    square_area = square_area.flatten()
    zm = resample_structured_grid(rx, ry, z, rxm, rym).flatten()
    if at_raster_values is None:
        at_values = np.unique(zm)
    else:
        at_values = np.atleast_1d(at_raster_values)
    DATA = {}
    DATA['value'] = at_values
    if at_raster_value_names is not None:
        DATA['value name'] = at_raster_value_names
    if secondary_constraint_filename is None:
        DATA['Area'] = np.zeros(len(at_values))
        for ic, value in enumerate(at_values):
            area_ix = np.flatnonzero(zm == value)
            DATA['Area'][ic] = np.sum(square_area[area_ix])
        DATA['Area percent'] = 100 * DATA['Area']/DATA['Area'].sum()
    else:
        rrx, rry, constraint = read_raster(secondary_constraint_filename)
        constraint = resample_structured_grid(rrx, rry, constraint, rxm, rym, interpmethod='nearest').flatten()
        if limit_constraint_range is not None:
            constraint = np.where((constraint >= np.min(limit_constraint_range)) & (
                constraint <= np.max(limit_constraint_range)), constraint, np.nan)
        for ic, rval in enumerate(np.unique(constraint)):
            if ~np.isnan(rval):
                zz = zm[constraint == rval]
                sqa = square_area[constraint == rval]
                rcolname = f'Area, constraint value {rval}'
                ccolname = f'Count, constraint value {rval}'
                DATA[rcolname] = np.zeros(len(at_values))
                DATA[ccolname] = np.zeros(len(at_values))
                for iic, value in enumerate(at_values):
                    area_ix = np.flatnonzero(zz == value)
                    DATA[ccolname][iic] = len(area_ix)
                    DATA[rcolname][iic] = np.sum(sqa[area_ix])
                DATA[f'Area percent, constraint value {rval}'] = 100 * \
                    DATA[rcolname]/DATA[rcolname].sum()
    return pd.DataFrame(DATA)

    # def calc_area_change(self, ofilename, crs, stylefile=None):
    #     """Export the areas of the given file. Find a UTM of the given crs and calculate in m2."""

    #     cfile = ofilename.replace(".tif", ".csv")
    #     if os.path.isfile(cfile):
    #         os.remove(cfile)

    #     # if stylefile is not None:
    #     #     sdf = df_from_qml(stylefile)

    #     # get the basename and use the raster in the instance to get the min / max
    #     basename = os.path.splitext(os.path.basename(ofilename))[0]
    #     raster = QgsProject.instance().mapLayersByName(basename)[0]

    #     xmin = raster.extent().xMinimum()
    #     xmax = raster.extent().xMaximum()
    #     ymin = raster.extent().yMinimum()
    #     ymax = raster.extent().yMaximum()

    #     # using the min and max make sure the crs doesn't change across grids
    #     if crs==4326:
    #         assert find_utm_srid(xmin, ymin, crs) == find_utm_srid(
    #             xmax,
    #             ymax,
    #             crs,
    #         ), "grid spans multiple utms"
    #         crs_found = find_utm_srid(xmin, ymin, crs)

    #         # create a temporary file for reprojection
    #         outfile = tempfile.NamedTemporaryFile(suffix=".tif").name
    #         # cmd = f'gdalwarp -s_srs EPSG:{crs} -t_srs EPSG:{crs_found} -r near -of GTiff {ofilename} {outfile}'
    #         # os.system(cmd)

    #         reproject_params = {
    #             "INPUT": ofilename,
    #             "SOURCE_CRS": QgsCoordinateReferenceSystem(f"EPSG:{crs}"),
    #             "TARGET_CRS": QgsCoordinateReferenceSystem(f"EPSG:{crs_found}"),
    #             "RESAMPLING": 0,
    #             "NODATA": None,
    #             "TARGET_RESOLUTION": None,
    #             "OPTIONS": "",
    #             "DATA_TYPE": 0,
    #             "TARGET_EXTENT": None,
    #             "TARGET_EXTENT_CRS": QgsCoordinateReferenceSystem(f"EPSG:{crs_found}"),
    #             "MULTITHREADING": False,
    #             "EXTRA": "",
    #             "OUTPUT": outfile,
    #         }

    #         # reproject to a UTM crs for meters calculation
    #         processing.run("gdal:warpreproject", reproject_params)

    #         params = {
    #             "BAND": 1,
    #             "INPUT": outfile,
    #             "OUTPUT_TABLE": cfile,
    #         }

    #         processing.run("native:rasterlayeruniquevaluesreport", params)
    #         # remove the temporary file
    #         os.remove(outfile)
    #     else:
    #         params = {
    #             "BAND": 1,
    #             "INPUT": ofilename,
    #             "OUTPUT_TABLE": cfile,
    #         }

    #         processing.run("native:rasterlayeruniquevaluesreport", params)

    #     df = pd.read_csv(cfile, encoding="cp1252")
    #     if "m2" in df.columns:
    #         df.rename(columns={"m2": "Area"}, inplace=True)
    #     elif "m²" in df.columns:
    #         df.rename(columns={"m²": "Area"}, inplace=True)
    #     elif "Unnamed: 2" in df.columns:
    #         df.rename(columns={"Unnamed: 2": "Area"}, inplace=True)
    #     df = df.groupby(by=["value"]).sum().reset_index()

    #     df["percentage"] = (df["Area"] / df["Area"].sum()) * 100.0

    #     df["value"] = df["value"].astype(float)
    #     # recode 0 to np.nan
    #     df.loc[df["value"] == 0, "value"] = float("nan")
    #     # sort ascending values
    #     df = df.sort_values(by=["value"])

    #     if stylefile is not None:
    #         df = pd.merge(df, sdf, how="left", on="value")
    #         df.loc[:, ["value", "label", "count", "Area", "percentage"]].to_csv(
    #             cfile,
    #             index=False,
    #         )
    #     else:
    #         df.loc[:, ["value", "count", "Area", "percentage"]].to_csv(
    #             cfile,
    #             na_rep="NULL",
    #             index=False,
    #             )