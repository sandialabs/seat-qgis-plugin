Model Results Directories
--------------------------------------

For a comprehensive analysis, you'll need to specify two distinct directories from your model results:
  1. **Baseline Results Directory**: Contains the model results without any devices.
  2. **Model Results Directory**: Houses the model results with devices incorporated.

.. figure:: ../../media/model_results_directory_input.webp
   :scale: 100 %
   :alt: Interface to choose the Model Results Directory in SEAT's GUI.

To proceed:

- Ensure you select the directory that contains the model results with devices, and another for the baseline results without devices.
- It's crucial that all model results are in the `netCDF` format.

There are two accepted formats for the model results:

  1. A single concatenated `*.nc` file with the model run number as its first dimension, arranged as [RunNum, Depth, X, Y]. Ensure there's only one file per directory when using this format.
  2. Multiple files, each named in the format: `name_RunNum_map.nc`. Here, `RunNum` should precede `map.nc` and be flanked by a single underscore on each side.

.. note::    
   The number of baseline model files should be identical to the number of device model files to maintain consistency in your analysis.
