#!/usr/bin/python

# Oregon Coast Delft3D WEC Difference Plotting

# Plot normalized comparison of simulations with WECs and without WECS for
# user selected variable for all boundary conditions

# Usage:
# python delft_wecs_diff_all_bcs.py

# Output: # of BCs figures saved to run_dir directory

# Revision Log
# -----------------------------------------------------------------------------
# 09/12/2017 Chris Flanary - created script
# 09/18/2017 CF - added file_num variable for multiple NetCDF files and CSV
#                 file read for run order
# 12/20/2017 Kaus added risk assessment code
# 12/22/2017 Kaus added creation of metric file for habitat polygons
# 06/09/2020 Kaus simplified script for a single run, for Matt
# 06/26/2020 Matt added GDAL geotransform
# 05/10/2022 Eben Pendleton added comments

import glob
import os

import numpy as np
import pandas as pd
from matplotlib.tri import LinearTriInterpolator, TriAnalyzer, Triangulation
from netCDF4 import Dataset
from osgeo import gdal, osr

# these imports currently don't work :(
# from qgis.core import *
# import qgis.utils


