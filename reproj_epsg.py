"""

Purpose: Transform raster by EPSG
Project: Sandia

Example usage:  reproj_epsg(infile, outfile)
2021-03-21. Initially Written. Eben Pendleton

"""

from pyproj import Transformer, Proj, transform

def reproj_epsg(x,y, espg_in, espg_out, outfile):

    """
    Given an input file and input espg and output espg the in file is reprojected to the out file

    :param x: Input file path
    :param espg_in: ESPG input file
    :param espg_out: ESPG output file
    :return: outfile path
    """
    transformer = Transformer.from_crs("EPSG:32606", "EPSG:4326", always_xy=True)
    
    return outfile
