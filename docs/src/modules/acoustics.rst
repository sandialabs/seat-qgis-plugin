
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
- GeoTIFF raster files: Visualize calculated paracoustic, calculated stressor, threshold exceeded receptor, species percent and species density.
    
    - Output layers are interpolated onto structured grids.
    - **calculated_stressor.tif** : The probability weight difference between with devices and baseline models results. 

        * for acoustics assumes baseline=0 if no baseline model files provided.

    - **receptor.tif** : the receptor file interpolated to the same grid as the output

    * **calculate_paracousti.tif** : the calculated with device probability weighted paracousti file.
    * **Threshold_exceeded_receptor.tif** : the percent of time the acoustic threshold was exceeded.
    * **species_percent.tif** : the threshold exceeded and weighted species percent.
    * **species_density.tif** : the threshold exceeded and weighted species density.

- CSV files: Contain statistics of area calculations for various layers.

  * The stressor values are binned into 25 bins and the surface area in which that change occurred, the percent of the overall model domain, and number of cells within the stressor is saved to a csv file.  
   
    + Lat/Lon converted to UTM (meter) coordinates for calculation.
    + UTM remains in the original unit of measure

- When a receptor is included, the stressor and stressor with receptor values are further segmented by unique receptor values.
  
  * For acoustics, the threshold exceeded, the species percent, and species density are generated.


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

