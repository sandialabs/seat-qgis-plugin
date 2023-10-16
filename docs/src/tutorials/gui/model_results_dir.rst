Model Results Directory
-----------------------

There are two model result directories needed for the analysis:
  1. Baseline results directory 
  2. Model results directory

.. figure:: ../../media/model_results_directory_input.webp
   :scale: 100 %
   :alt: Model Results Directory

Select the directory containing the model results with devices and the directory containing the baseline (without devices) model results. These model results must be in netCDF format.

The model results can have 2 formats.
  
  1. A concatenated single \*.nc file with the model run number as the first dimension [RunNum, Depth, X, Y]. (Only one file per directory if this format is used).
  2. Multiple files with naming format name_RunNum_map.nc, where RunNum must come before the map.nc and be separated by a single underscore on either side.

The number of baseline model files must match the device model files.