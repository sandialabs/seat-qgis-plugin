Model Results Directories
--------------------------------------

For a comprehensive analysis, you'll need to specify two distinct directories from your model results:
  1. **Baseline Results Directory**: Contains the model results without any devices. If left blank, a 0dB baseline will be assumed.
  2. **Model Results Directory**: Houses the model results with devices incorporated.

.. figure:: ../../media/model_results_directory_input.webp
   :scale: 100 %
   :alt: Interface to choose the Model Results Directory in SEAT's GUI.

To proceed:

- Ensure you select the directory that contains the model results with devices, and another for the baseline results without devices.
- It's crucial that all model results are in the `netCDF` format.

The accepted formats for the model results:

  1. One `.nc` file per probability.

.. note::    
   The number and filename of baseline model files should be identical to the number of device model files to maintain consistency in your analysis.
