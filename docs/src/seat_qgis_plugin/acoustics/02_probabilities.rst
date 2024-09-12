Probabilities 
------------------------

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
