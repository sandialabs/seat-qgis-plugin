import numpy as np
import pandas as pd
from matplotlib.tri import LinearTriInterpolator, TriAnalyzer, Triangulation
from scipy.interpolate import griddata
from osgeo import gdal, osr
import os
from .Find_UTM_srid import find_utm_srid # UTM finder

def estimate_grid_spacing(x,y, nsamples=100):
    import random
    import sys
    coords = list(set(zip(x,y)))
    if nsamples != len(x):
        points = [random.choice(coords) for i in range(nsamples)] # pick N random points
    else:
        points = coords
    MD = []
    for p0x, p0y in points:
        minimum_distance = sys.maxsize
        for px, py in coords:
            distance = np.sqrt((p0x - px) ** 2 + (p0y - py) ** 2)
            if (distance < minimum_distance) & (distance !=0):
                minimum_distance = distance
        MD.append(minimum_distance)
    dxdy = np.median(MD)
    return dxdy

def create_structured_array_from_unstructured(x, y, z, dxdy, flatness=0.2):
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

def redefine_structured_grid(x,y,z):
    min_x = np.nanmin(x)
    min_y = np.nanmin(y)
    max_x = np.nanmax(x)
    max_y = np.nanmax(y)
    xx = np.linspace(min_x, max_x, x.shape[1])
    yy = np.linspace(min_y, max_y, y.shape[0])
    dx = np.nanmin([np.nanmedian(np.diff(xx)), np.nanmedian(np.diff(yy))])
    xx = np.arange(min_x, max_x+dx, dx)
    yy = np.arange(min_y, max_y+dx, dx)
    x_new, y_new = np.meshgrid(xx,yy)
    z_new = griddata((x.flatten(), y.flatten()), z.flatten(), (x_new, y_new), method='nearest', fill_value=0)
    return x_new, y_new, z_new

def calc_receptor_array(receptor_filename, x, y, latlon=False):
    # if ((receptor_filename is not None) or (not receptor_filename == "")):
    if not((receptor_filename is None) or (receptor_filename == "")):
        if receptor_filename.endswith('.tif'):
            data = gdal.Open(receptor_filename)
            img = data.GetRasterBand(1)
            receptor_array = img.ReadAsArray()
            receptor_array[receptor_array < 0] = 0
            (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = data.GetGeoTransform()
            cols = data.RasterXSize
            rows = data.RasterYSize
            r_rows = np.arange(rows) * y_size + upper_left_y + (y_size / 2)
            r_cols = np.arange(cols) * x_size + upper_left_x + (x_size / 2)
            if latlon==True:
                r_cols = np.where(r_cols<0, r_cols+360, r_cols)
            x_grid, y_grid = np.meshgrid(r_cols, r_rows)
            receptor_array = griddata((x_grid.flatten(), y_grid.flatten()), receptor_array.flatten(), (x,y), method='nearest', fill_value=0)

        elif receptor_filename.endswith('.csv'):
            receptor_array = pd.read_csv(receptor_filename, header=None, index_col=0).to_numpy().item() * np.ones(x.shape)
        else:
            raise Exception("Invalid Recetpor File Type. Must be of type .tif or .csv")
    else:
        # taucrit without a receptor
        #Assume the following grain sizes and conditions for typical beach sand (Nielsen, 1992 p.108)
        receptor_array = 200*1e-6 * np.ones(x.shape)
    return receptor_array

def trim_zeros(x,y,z1, z2):
    #edges of structured array have zeros, not sure if universal issue
    return x[1:-1,1:-1], y[1:-1,1:-1], z1[:, :, 1:-1, 1:-1], z2[:, :, 1:-1, 1:-1]



def create_raster(
    output_path,
    cols,
    rows,
    nbands,
):

    """Create a gdal raster object."""
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
):

    """Create the output raster."""
    # create output raster
    # (upper_left_x, x_resolution, x_skew 0, upper_left_y, y_skew 0, y_resolution).
    # Need to rotate to go from np array to geo tiff. This can vary depending on the methods used above. Will need to test for this.
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
    # output_band.SetNoDataValue(no_data) #Not an issue, may be in other cases?
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

    :param lon: WGS 84 (srid 4326) longitude value)
    :param lat: WGS 84 (srid 4326) latitude value)
    :param srid: WGS 84 srid 4326 to make sure the function is appropriate
    :return: out_srid: UTM srid
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

def calculate_grid_sqarea_latlon2m(rx, ry):
    import numpy as np
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    rxx = np.where(rx>180, rx-360, rx)
    lon2D,lat2D = rxx, ry
    _,_, distEW = geod.inv(lon2D[:,:-1],lat2D[:,1:], lon2D[:,1:], lat2D[:,1:])
    _,_, distNS = geod.inv(lon2D[1:,:],lat2D[1:,:], lon2D[1:,:], lat2D[:-1,:])
    square_area = distEW[1:,:] * distNS[:,1:]
    np.nanmean(square_area)
    rxm = np.zeros(square_area.shape)
    rym = np.zeros(square_area.shape)
    for row in range(rx.shape[0]-1):
        rxm[row,:] = (rx[row,:-1]+ rx[row,1:])/2
    for col in range(ry.shape[1]-1):
        rym[:,col] = (ry[:-1,col]+ ry[1:,col])/2
    return rxm, rym, square_area