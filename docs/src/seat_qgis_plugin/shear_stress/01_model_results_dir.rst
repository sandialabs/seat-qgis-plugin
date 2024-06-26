
Model Results Directories
--------------------------------------

For a comprehensive analysis, you'll need to specify two distinct directories from your model results:
  1. **Baseline Model Results Directory**: Contains the model results **without** devices.
  2. **Device Model Results Directory**: Contains the model results **with** devices.

.. figure:: ../../media/model_results_directory_input.webp
   :scale: 100 %
   :alt: Interface to choose the Model Results Directory in SEAT's GUI.


All the results must be in ``netcdf`` format. 
Multiple run outputs can be structured in the following ways:

  1. A single `*.nc` file with the model run number as its first dimension, arranged as [RunNum, Depth, X, Y]. Ensure there's only one file per directory when using this format.

  2. Multiple files, each named in the format: `name_RunNum_map.nc`. 
      - `name` is the common prefix for all files.
      - `RunNum` is the model run number.
      - `map` is the map name. 
      - The files should be arranged in the following structure:
        - `name_1_map.nc`
        - `name_2_map.nc`
        - `name_3_map.nc`

.. note::    
   The number of baseline model files should be identical to the number of device model files to maintain consistency in your analysis.
