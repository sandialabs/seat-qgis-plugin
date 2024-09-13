# pylint: disable=too-many-statements
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
"""
velocity_module.py: Definitions for stressor modules.

This module includes functions for estimating grid spacing, creating structured
arrays from unstructured data, redefining grids, resampling grids, and handling
raster data.

Dependencies:
- pyproj, osgeo, numpy, pandas, matplotlib, scipy
"""
import os
import sys
import random
from typing import List, Tuple, Dict
import numpy as np
from numpy.typing import NDArray
from pyproj import Geod
import pandas as pd
from pandas import DataFrame
from matplotlib.tri import LinearTriInterpolator, TriAnalyzer, Triangulation
from scipy.interpolate import griddata
from osgeo import gdal, osr


def estimate_grid_spacing(
    x: NDArray[np.float64], y: NDArray[np.float64], nsamples: int = 100
) -> float:
    """
    Estimate grid spacing for an unstructured grid to create a structured grid.

    Parameters
    ----------
    x, y : array
        Coordinates of the points.
    nsamples : int, optional
        Number of random points to sample. Default is 100.

    Returns
    -------
    float
        Estimated median spacing between samples.
    """
    random.seed(10)
    coords = list(set(zip(x, y)))
    if nsamples != len(x):
        points = [
            random.choice(coords) for i in range(nsamples)
        ]  # pick N random points
    else:
        points = coords
    md = []
    for p0x, p0y in points:
        minimum_distance = sys.maxsize
        for px, py in coords:
            distance = np.sqrt((p0x - px) ** 2 + (p0y - py) ** 2)
            if (distance < minimum_distance) & (distance != 0):
                minimum_distance = distance
        md.append(minimum_distance)
    dxdy = np.median(md)
    return dxdy


def create_structured_array_from_unstructured(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    dxdy: float,
    flatness: float = 0.2,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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
    refx = np.arange(np.nanmin(x), np.nanmax(x) + dxdy, dxdy)
    refy = np.arange(np.nanmin(y), np.nanmax(y) + dxdy, dxdy)
    refxg, refyg = np.meshgrid(refx, refy)
    tri = Triangulation(x, y)
    mask = TriAnalyzer(tri).get_flat_tri_mask(flatness)
    tri.set_mask(mask)
    tli = LinearTriInterpolator(tri, z)
    z_interp = tli(refxg, refyg)
    return refxg, refyg, z_interp.data


def redefine_structured_grid(
    x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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
    xx = np.arange(min_x, max_x + dx, dx)
    yy = np.arange(min_y, max_y + dx, dx)
    x_new, y_new = np.meshgrid(xx, yy)
    z_new = griddata(
        (x.flatten(), y.flatten()),
        z.flatten(),
        (x_new, y_new),
        method="nearest",
        fill_value=0,
    )
    return x_new, y_new, z_new


def resample_structured_grid(
    x_grid: NDArray[np.float64],
    y_grid: NDArray[np.float64],
    z: NDArray[np.float64],
    x_grid_out: NDArray[np.float64],
    y_grid_out: NDArray[np.float64],
    interpmethod: str = "nearest",
) -> NDArray[np.float64]:
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
    x_grid_out : array
        x-coordinates.
    y_grid_out : array
        y-coordinates.
    interpmethod : str, optional
        interpolation method to use. The default is 'nearest'.

    Returns
    -------
    array
        interpolated z-value on X/Y _grid_out.

    """
    return griddata(
        (x_grid.flatten(), y_grid.flatten()),
        z.flatten(),
        (x_grid_out, y_grid_out),
        method=interpmethod,
        fill_value=0,
    )


def calc_receptor_array(
    receptor_filename: str,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    latlon: bool = False,
    mask: NDArray[np.bool_] = None,
) -> NDArray[np.float64]:
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
        if receptor_filename.endswith(".tif"):
            data = gdal.Open(receptor_filename)
            img = data.GetRasterBand(1)
            receptor_array = img.ReadAsArray()
            receptor_array[receptor_array < 0] = 0
            (upper_left_x, x_size, _, upper_left_y, _, y_size) = data.GetGeoTransform()
            cols = data.RasterXSize
            rows = data.RasterYSize
            r_rows = np.arange(rows) * y_size + upper_left_y + (y_size / 2)
            r_cols = np.arange(cols) * x_size + upper_left_x + (x_size / 2)
            if latlon:
                r_cols = np.where(r_cols < 0, r_cols + 360, r_cols)
            x_grid, y_grid = np.meshgrid(r_cols, r_rows)
            receptor_array = griddata(
                (x_grid.flatten(), y_grid.flatten()),
                receptor_array.flatten(),
                (x, y),
                method="nearest",
                fill_value=0,
            )

        elif receptor_filename.endswith(".csv"):
            receptor_array = pd.read_csv(
                receptor_filename, header=None, index_col=0
            ).to_numpy().item() * np.ones(x.shape)
        else:
            raise ValueError(
                f"Invalid Receptor File {receptor_filename}. Must be of type .tif or .csv"
            )
    else:
        # taucrit without a receptor
        # Assume the following grain sizes and conditions for
        # typical beach sand (Nielsen, 1992 p.108)
        receptor_array = 200 * 1e-6 * np.ones(x.shape)
    if mask is not None:
        receptor_array = np.where(mask, receptor_array, np.nan)
    return receptor_array


def trim_zeros(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z1: NDArray[np.float64],
    z2: NDArray[np.float64],
) -> Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
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
    output_path: str, cols: int, rows: int, nbands: int, e_type: int = gdal.GDT_Float32
) -> gdal.Dataset:
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
    e_type : gdal, optional
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
        e_type,
    )

    # spatial_reference = osr.SpatialReference()
    # spatial_reference.ImportFromEPSG(spatial_reference_system_wkid)
    # output_raster.SetProjection(spatial_reference.ExportToWkt())

    # returns gdal data source raster object
    return output_raster


def numpy_array_to_raster(
    output_raster: gdal.Dataset,
    numpy_array: NDArray[np.float64],
    bounds: Tuple[float, float],
    cell_resolution: Tuple[float, float],
    spatial_reference_system_wkid: int,
    output_path: str,
    nodata_val: float = None,
) -> str:
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
        output_band.SetNoDataValue(nodata_val)  # Not an issue, may be in other cases?
    output_band.WriteArray(numpy_array)

    output_band.FlushCache()
    # You want this false, true will make computed results, but is
    #  faster, could be a setting in the UI perhaps, esp for large rasters?
    output_band.ComputeStatistics(
        False,
    )

    if not os.path.exists(output_path):
        raise RuntimeError(f"Failed to create raster: {output_path}")
    # this closes the file
    output_raster = None
    return output_path


def find_utm_srid(lon: float, lat: float, srid: int) -> int:
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


def read_raster(
    raster_name: str,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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
    (upper_left_x, x_size, _, upper_left_y, _, y_size) = data.GetGeoTransform()
    cols = data.RasterXSize
    rows = data.RasterYSize
    r_rows = np.arange(rows) * y_size + upper_left_y + (y_size / 2)
    r_cols = np.arange(cols) * x_size + upper_left_x + (x_size / 2)
    rx, ry = np.meshgrid(r_cols, r_rows)
    data = None

    return rx, ry, raster_array


def secondary_constraint_geotiff_to_numpy(
    filename: str,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Converts a secondary constraint GeoTIFF file to a NumPy array.

    Parameters
    ----------
    filename : str
        The file path to the GeoTIFF file.

    Returns
    -------
    tuple
        A tuple containing:
        - x_grid (ndarray): The X coordinates grid.
        - y_grid (ndarray): The Y coordinates grid.
        - array (ndarray): The data values from the GeoTIFF file as a 2D array.
    """
    data = gdal.Open(filename)
    img = data.GetRasterBand(1)
    array = img.ReadAsArray()

    (upper_left_x, x_size, _, upper_left_y, _, y_size) = data.GetGeoTransform()
    cols = data.RasterXSize
    rows = data.RasterYSize
    r_rows = np.arange(rows) * y_size + upper_left_y + (y_size / 2)
    r_cols = np.arange(cols) * x_size + upper_left_x + (x_size / 2)
    prj = data.GetProjection()
    srs = osr.SpatialReference(wkt=prj)
    if srs.GetAttrValue("AUTHORITY", 1) == "4326":
        r_cols = np.where(r_cols < 0, r_cols + 360, r_cols)
    x_grid, y_grid = np.meshgrid(r_cols, r_rows)
    return x_grid, y_grid, array


def calculate_cell_area(
    rx: NDArray[np.float64], ry: NDArray[np.float64], latlon: bool = True
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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

    if latlon:
        geod = Geod(ellps="WGS84")
        lon_2d, lat_2d = np.where(rx > 180, rx - 360, rx), ry
        _, _, dist_ew = geod.inv(
            lon_2d[:, :-1], lat_2d[:, 1:], lon_2d[:, 1:], lat_2d[:, 1:]
        )
        _, _, dist_ns = geod.inv(
            lon_2d[1:, :], lat_2d[1:, :], lon_2d[1:, :], lat_2d[:-1, :]
        )
        square_area = dist_ew[1:, :] * dist_ns[:, 1:]
    else:
        square_area = np.zeros((rx.shape[0] - 1, ry.shape[1] - 1))
        for row in range(rx.shape[0] - 1):
            for col in range(ry.shape[1] - 1):
                dx = rx[row, col + 1] - rx[row, col]
                dy = ry[row + 1, col] - ry[row, col]
                square_area[row, col] = dx * dy
    rxm = np.zeros(square_area.shape)
    rym = np.zeros(square_area.shape)
    for row in range(rx.shape[0] - 1):
        rxm[row, :] = (rx[row, :-1] + rx[row, 1:]) / 2
    for col in range(ry.shape[1] - 1):
        rym[:, col] = (ry[:-1, col] + ry[1:, col]) / 2

    return rxm, rym, square_area


def bin_data(
    zm: NDArray[np.float64], square_area: NDArray[np.float64], nbins: int = 25
) -> Dict[str, NDArray[np.float64]]:
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
    data : Dictionary
        Dictionary for each bin contating
            bin start : the starting value of each bin
            bin end : the last value of each bin
            bin center: the center value of each bin
            count :number of cells in each bin
            Area : area overwhich the binned values occur

    """
    hist, bins = np.histogram(zm, bins=nbins)
    center = (bins[:-1] + bins[1:]) / 2
    data = {}
    data["bin start"] = bins[:-1]
    data["bin end"] = bins[1:]
    data["bin center"] = center
    data["count"] = hist
    data["Area"] = np.zeros(hist.shape)
    for ic, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
        if ic < len(hist) - 1:
            area_ix = np.flatnonzero((zm >= bin_start) & (zm < bin_end))
        else:
            area_ix = np.flatnonzero((zm >= bin_start) & (zm <= bin_end))
        data["Area"][ic] = np.sum(square_area[area_ix])
    return data


def bin_receptor(
    zm: NDArray[np.float64],
    receptor: NDArray[np.float64],
    square_area: NDArray[np.float64],
    nbins: int = 25,
    receptor_names: List[str] = None,
    receptor_type: str = "receptor",
) -> Dict[str, NDArray[np.float64]]:
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
    receptor_type : str, optional
        name to display in output (eg. grain size, risk layer). the default is receptor.

    Returns
    -------
    data : Dictionary
        Dictionary with keys corresponding to each unique receptor value
        each containing for each bin

            bin start : the starting value of each bin
            bin end : the last value of each bin
            bin center: the center value of each bin
            count :number of cells in each bin
            Area : area overwhich the binned values occur.
            Area percent : percent of the domain overwhich the binned values occur.

    """
    hist, bins = np.histogram(zm, bins=nbins)
    center = (bins[:-1] + bins[1:]) / 2
    data = {}
    data["bin start"] = bins[:-1]
    data["bin end"] = bins[1:]
    data["bin center"] = center
    for ic, rval in enumerate(np.unique(receptor)):
        zz = zm[receptor == rval]
        sqa = square_area[receptor == rval]
        rcolname = (
            f"Area, {receptor_type} value {rval}"
            if receptor_names is None
            else receptor_names[ic]
        )
        data[rcolname] = np.zeros(hist.shape)
        for ic, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
            if ic < len(hist) - 1:
                area_ix = np.flatnonzero((zz >= bin_start) & (zz < bin_end))
            else:
                area_ix = np.flatnonzero((zz >= bin_start) & (zz <= bin_end))
            data[rcolname][ic] = np.sum(sqa[area_ix])
        data[f"Area percent, {receptor_type} value {rval}"] = (
            100 * data[rcolname] / data[rcolname].sum()
        )
    return data


def bin_layer(
    raster_filename: str,
    receptor_filename: str = None,
    receptor_names: List[str] = None,
    limit_receptor_range: Tuple[float, float] = None,
    latlon: bool = True,
    receptor_type: str = "receptor",
) -> DataFrame:
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
    receptor_type : str, optional
        name to display in output (eg. grain size, risk layer). the default is receptor.

    Returns
    -------
    DataFrame
        DataFrame with bins as rows and statistics values as columns.
            [bin stats, area, count, percent]

    """
    rx, ry, z = read_raster(raster_filename)
    rxm, rym, square_area = calculate_cell_area(rx, ry, latlon)
    square_area = square_area.flatten()
    zm = resample_structured_grid(rx, ry, z, rxm, rym, interpmethod="linear").flatten()
    if receptor_filename is None:
        data = bin_data(
            zm[np.invert(np.isnan(zm))], square_area[np.invert(np.isnan(zm))], nbins=25
        )
        # DF = pd.DataFrame(data)
        data["Area percent"] = 100 * data["Area"] / data["Area"].sum()
    else:
        rrx, rry, receptor = read_raster(receptor_filename)
        receptor = resample_structured_grid(rrx, rry, receptor, rxm, rym).flatten()
        if limit_receptor_range is not None:
            receptor = np.where(
                (receptor >= np.min(limit_receptor_range))
                & (receptor <= np.max(limit_receptor_range)),
                receptor,
                0,
            )
        data = bin_receptor(
            zm[np.invert(np.isnan(zm))],
            receptor[np.invert(np.isnan(zm))],
            square_area[np.invert(np.isnan(zm))],
            receptor_names=receptor_names,
            receptor_type=receptor_type,
        )
    return pd.DataFrame(data)


def classify_layer_area(
    raster_filename: str,
    receptor_filename: str = None,
    at_values: List[float] = None,
    value_names: List[str] = None,
    limit_receptor_range: Tuple[float, float] = None,
    latlon: bool = True,
    receptor_type: str = "receptor",
) -> DataFrame:
    """
    Creates a dataframe of raster values and associtaed area and
    percent of array at specified raster values.

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
    receptor_type : str, optional
        name to display in output (eg. grain size, risk layer). the default is receptor.


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
    data = {}
    data["value"] = at_values
    if value_names is not None:
        data["value name"] = value_names
    if receptor_filename is None:
        data["Area"] = np.zeros(len(at_values))
        for ic, value in enumerate(at_values):
            area_ix = np.flatnonzero(zm == value)
            data["Area"][ic] = np.sum(square_area[area_ix])
        data["Area percent"] = 100 * data["Area"] / data["Area"].sum()
    else:
        rrx, rry, receptor = read_raster(receptor_filename)
        if limit_receptor_range is not None:
            receptor = np.where(
                (receptor >= np.min(limit_receptor_range))
                & (receptor <= np.max(limit_receptor_range)),
                receptor,
                0,
            )
        receptor = resample_structured_grid(rrx, rry, receptor, rxm, rym).flatten()
        for ic, rval in enumerate(np.unique(receptor)):
            zz = zm[receptor == rval]
            sqa = square_area[receptor == rval]
            rcolname = f"Area, {receptor_type} value {rval}"
            ccolname = f"Count, {receptor_type} value {rval}"
            data[rcolname] = np.zeros(len(at_values))
            data[ccolname] = np.zeros(len(at_values))
            for iic, value in enumerate(at_values):
                area_ix = np.flatnonzero(zz == value)
                data[ccolname][iic] = len(area_ix)
                data[rcolname][iic] = np.sum(sqa[area_ix])
            data[f"Area percent, {receptor_type} value {rval}"] = (
                100 * data[rcolname] / data[rcolname].sum()
            )
    return pd.DataFrame(data)


def classify_layer_area_2nd_constraint(
    raster_to_sample: str,
    secondary_constraint_filename: str,
    at_raster_values: List[float],
    at_raster_value_names: List[str],
    limit_constraint_range: Tuple[float, float] = None,
    latlon: bool = True,
    receptor_type: str = "receptor",
) -> DataFrame:
    """
    Classifies layer areas based on a secondary constraint raster.

    This function calculates the area of different classifications in a raster
    and applies an additional filter or constraint using a secondary raster file.

    Parameters
    ----------
    raster_to_sample : str
        Path to the raster file to be sampled.
    secondary_constraint_filename : str or None
        Path to the secondary constraint raster file. If None, no secondary constraint is applied.
    at_raster_values : list or None
        List of values in the raster to classify. If None, all unique values are considered.
    at_raster_value_names : list or None
        List of names corresponding to the `at_raster_values` for more descriptive output.
    limit_constraint_range : tuple or None, optional
        A tuple specifying the range (min, max) to limit the secondary constraint values.
        Default is None.
    latlon : bool, optional
        Boolean to indicate if the coordinate system is latitude/longitude. Default is True.
    receptor_type : str, optional
        Type of receptor for naming purposes in the output. Default is "receptor".

    Returns
    -------
    pd.DataFrame
        A DataFrame with areas and their percentages calculated for each classification
        and optionally for each classification within a constraint range.
    """
    rx, ry, z = read_raster(raster_to_sample)
    rxm, rym, square_area = calculate_cell_area(rx, ry, latlon=latlon)
    square_area = square_area.flatten()
    zm = resample_structured_grid(rx, ry, z, rxm, rym).flatten()
    if at_raster_values is None:
        at_values = np.unique(zm)
    else:
        at_values = np.atleast_1d(at_raster_values)
    data = {}
    data["value"] = at_values
    if at_raster_value_names is not None:
        data["value name"] = at_raster_value_names
    if secondary_constraint_filename is None:
        data["Area"] = np.zeros(len(at_values))
        for ic, value in enumerate(at_values):
            area_ix = np.flatnonzero(zm == value)
            data["Area"][ic] = np.sum(square_area[area_ix])
        data["Area percent"] = 100 * data["Area"] / data["Area"].sum()
    else:
        rrx, rry, constraint = read_raster(secondary_constraint_filename)
        constraint = resample_structured_grid(
            rrx, rry, constraint, rxm, rym, interpmethod="nearest"
        ).flatten()
        if limit_constraint_range is not None:
            constraint = np.where(
                (constraint >= np.min(limit_constraint_range))
                & (constraint <= np.max(limit_constraint_range)),
                constraint,
                np.nan,
            )
        for ic, rval in enumerate(np.unique(constraint)):
            if ~np.isnan(rval):
                zz = zm[constraint == rval]
                sqa = square_area[constraint == rval]
                rcolname = f"Area, {receptor_type} value {rval}"
                ccolname = f"Count, {receptor_type} value {rval}"
                data[rcolname] = np.zeros(len(at_values))
                data[ccolname] = np.zeros(len(at_values))
                for iic, value in enumerate(at_values):
                    area_ix = np.flatnonzero(zz == value)
                    data[ccolname][iic] = len(area_ix)
                    data[rcolname][iic] = np.sum(sqa[area_ix])
                data[f"Area percent, {receptor_type} value {rval}"] = (
                    100 * data[rcolname] / data[rcolname].sum()
                )
    return pd.DataFrame(data)
