
Acoustics Module
----------------

The `acoustics_module.py` is a component of the SEAT aimed at assessing the impact of acoustic signal changes from paracoustic files. This module facilitates the understanding and visualization of how different acoustic variables alter in aquatic environments, especially when devices are present.

Data
^^^^

Input 
""""""
- NetCDF files: Contain acoustic data for scenarios with and without devices present.
- Optional:

  - Receptor file: Contains threshold values for acoustic variables.
  - Probability/Boundary Condition file: Used to weight different run scenarios.
  - Species files: Contains density or percent data of species.

Output 
""""""
- Output is saved in the **Acoustics Module** subdirectory.
- GeoTIFF raster files: Visualize calculated paracoustic, calculated stressor, threshold exceeded receptor, species percent and species density.
    
  - Output layers are interpolated onto structured grids.
  - **paracousti_stressor.tif** : The probability weight difference between with devices and baseline models results. 
  
    * for acoustics assumes baseline=0 if no baseline model files provided.
  
  - **paracousti_with_devices.tif**: The probability weighted signal with devices
  - **paracousti_without_devices.tif**: The probability weighted signal without devices (baseline)
  - **species_threshold_exceeded.tif** : the percent of time the acoustic threshold was exceeded.
  - **species_percent.tif** : the threshold exceeded and weighted species percent.
  - **species_density.tif** : the threshold exceeded and weighted species density.

- CSV files: Contain statistics of area calculations for various layers.
    + Lat/Lon converted to UTM (meter) coordinates for calculation.
    + UTM remains in the original unit of measure

  * The stressor values are binned into 25 bins and the surface area in which that change occurred, the percent of the overall model domain, and number of cells within the stressor is saved to a csv file.  
    - Output includes:

      - **paracousti_stressor.csv**
      - **paracousti_with_devices.csv**
      - **paracousti_without_devices.csv**
      - **species_threshold_exceeded.csv**
      - **species_percent.csv**
      - **species_density.csv**

    - When a risk layer receptor is included, the values are further segmented by unique risk layer values.
    - Output includes:

      - **paracousti_stressor_at_paracousti_risk_layer.csv**
      - **species_density_at_paracousti_risk_layer.csv**
      - **species_percent_at_paracousti_risk_layer.csv**
  

Sources
"""""""

Default
+++++++

The acoustics module is designed for paracousti data sources (https://sandialabs.github.io/Paracousti/). 

Acoustics variables:

- The variable is specified in the receptor file to allow for various weighting, sound pressure level, or sound exposure level thresholds. 

Coordinates are determined from the variable attributes

Alternative
+++++++++++

The Acoustics module can utilize alternate datasets with the following requirements:

- Variable name must be specified in the receptor file.
- Variable attributes must include the coordinates variable names, such that:

  * xcor, ycor = netcdf_dataset.variables[variable].coordinates.split() 

- The coordinates units attribute must include “degrees” if the coordinates are lat/lon such that:

  * 'degrees’ in ds.variables[<xcor variable>].units is True for lat/lon


Core Functions:
^^^^^^^^^^^^^^^

+--------------------------------------------+------------------------------------------------------------------+
| Function                                   | Description                                                      |
+============================================+==================================================================+
| ``create_species_array()``                 | Interpolates or creates an array of percent or density of species|
|                                            | from input files and coordinates.                                |
+--------------------------------------------+------------------------------------------------------------------+
| ``calculate_acoustic_stressors()``         | Calculates the stressor layers as arrays from model and parameter|
|                                            | input.                                                           |
+--------------------------------------------+------------------------------------------------------------------+
| ``run_acoustics_stressor()``               | Creates GeoTIFFs and area change statistics files for acoustic   |
|                                            | stressor change.                                                 |
+--------------------------------------------+------------------------------------------------------------------+
| ``redefine_structured_grid()``             | (From `stressor_utils`) Redefines grids to regular spacing, used |
|                                            | in `calculate_acoustic_stressors`.                               |
+--------------------------------------------+------------------------------------------------------------------+
| ``resample_structured_grid()``             | (From `stressor_utils`) Resamples grids, used in                 |
|                                            | `calculate_acoustic_stressors`.                                  |
+--------------------------------------------+------------------------------------------------------------------+

