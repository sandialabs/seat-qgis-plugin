Inputs
-------

Model Results Directories
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


For a comprehensive analysis, the SEAT QGIS plugin uses two distinct directories from your model results:
  1. **Baseline Results Directory**: Contains the model results without any devices. If left blank, a 0dB baseline will be assumed.
  2. **Model Results Directory**: Contains the model results with devices.

.. figure:: ../../media/model_results_directory_input.webp
   :scale: 100 %
   :alt: Interface to choose the Model Results Directory in SEAT's GUI.

The accepted formats for the model results:

  1. Multiple files, each named in the format: `name_condition.nc`. 
      - `name` is the common prefix for all files.
      - `run` is the run scenario designated the type of environmental condition (e.g., Hw0.5)
      - The files should be arranged in the following structure:
        - `name_run1.nc`
        - `name_run2.nc`
        - `name_run3.nc`

.. note::    
  - All model results are in the `netCDF` format.
  - The number and filename of baseline model files should be identical to the number of device model files to maintain consistency in your analysis.


Probabilities 
^^^^^^^^^^^^^^^

The probabilities file defines the likelihood of each model condition occurring. Both shear and stress velocity have probability files, but the format is different for acoustics than the shear and stress velocity modules.
Adhere to the prescribed naming convention (as delineated in the device/baseline model section). 
Note that this file correlates with the return interval in years. 


.. figure:: ../../media/probabilities_input.webp
   :scale: 100 %
   :alt: Interface depicting the Probabilities Input in SEAT's GUI.



**File Specifications**:

- If you're using .csv for the Species Percent Occurrence and Species Density Files, they must contain the essential columns: "latitude", "longitude", and either "percent" and/or "density". All supplementary columns will be overlooked.
- If you opt for a .tif format for the aforementioned files, ensure consistency in the EPSG code across them.

**Example of a Probabilities Input**

.. code-block:: text
   :caption: boundary_conditions.csv
  Paracousti File,Species Percent Occurance File,Species Density File,% of yr
  pacwave_3DSPLs_Hw0.5.nc,whale_watch_predictions_2021_01.csv,whale_watch_predictions_2021_01.csv,0
  pacwave_3DSPLs_Hw1.0.nc,whale_watch_predictions_2021_02.csv,whale_watch_predictions_2021_02.csv,2.729
  pacwave_3DSPLs_Hw1.5.nc,whale_watch_predictions_2021_03.csv,whale_watch_predictions_2021_03.csv,20.268
  pacwave_3DSPLs_Hw2.0.nc,whale_watch_predictions_2021_04.csv,whale_watch_predictions_2021_04.csv,39.769
  pacwave_3DSPLs_Hw2.5.nc,whale_watch_predictions_2021_05.csv,whale_watch_predictions_2021_05.csv,13.27
  pacwave_3DSPLs_Hw3.0.nc,whale_watch_predictions_2021_06.csv,whale_watch_predictions_2021_06.csv,3.49
  pacwave_3DSPLs_Hw3.5.nc,whale_watch_predictions_2021_07.csv,whale_watch_predictions_2021_07.csv,11.212
  pacwave_3DSPLs_Hw4.0.nc,whale_watch_predictions_2021_08.csv,whale_watch_predictions_2021_08.csv,0.593
  pacwave_3DSPLs_Hw4.5.nc,whale_watch_predictions_2021_09.csv,whale_watch_predictions_2021_09.csv,1.813
  pacwave_3DSPLs_Hw5.0.nc,whale_watch_predictions_2021_10.csv,whale_watch_predictions_2021_10.csv,6.462
  pacwave_3DSPLs_Hw5.5.nc,whale_watch_predictions_2021_11.csv,whale_watch_predictions_2021_11.csv,0
  pacwave_3DSPLs_Hw6.0.nc,whale_watch_predictions_2021_12.csv,whale_watch_predictions_2021_12.csv,0
  pacwave_3DSPLs_Hw6.5.nc,whale_watch_predictions_2021_01.csv,whale_watch_predictions_2021_01.csv,0
  pacwave_3DSPLs_Hw7.0.nc,whale_watch_predictions_2021_02.csv,whale_watch_predictions_2021_02.csv,0.086



Key:

- `ParAcousti File`: The name of the ParAcousti .nc file.
- `Species % Occurrence File`: Either a .csv or .tif file indicating species percent occurrence.
- `Species Density File`: Either a .csv or .tif file detailing species density.
- `% of yr`: Represents the percentage of the year.

Risk layer (Optional)
^^^^^^^^^^^^^^^^^^^^^^

The risk layer is a receptor file that serves as an additional input to each module and designated which layers are sensitive and would be effected by the acoustics. 
It must be a numerically classified .tif format. It is the same as what is used in the shear stress and velocity modules.

.. figure:: ../../media/risk_layer_gui_input.webp
   :scale: 100 %
   :alt: Risk Layer File

Represents a layer to evalute change against. Examples include vegetation habitat, marine ecosystems, contaminated sediments, marine protected areas, or archaeological artifacts.

- **File Type**: Supports geotiff (.tif) file format.
  
  - **Geotiff Details**:

    - Must have the same projection and datum as the model files.
    - Will be nearest-neighbor interpolated to align with the model files' grid points (structured/unstructured).
    - Must be integer classified eg. (0 = 'Kelp', 1 = 'Rock')