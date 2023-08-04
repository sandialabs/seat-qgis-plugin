import numpy as np
import pandas as pd
from matplotlib.tri import LinearTriInterpolator, TriAnalyzer, Triangulation
from scipy.interpolate import griddata
from osgeo import gdal, osr

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