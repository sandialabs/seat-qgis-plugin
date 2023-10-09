
Shear Stress Module
-------------------

The `shear_stress_module.py` a component of SEAT. It's aimed at assessing the impact of shear stressors on sediment mobility in aquatic environments. This module provides insights into how various devices or conditions can affect shear stress by allowing comparison between scenarios with and without these elements present.


Data
^^^^
Input 
""""""
- NetCDF files: Contain shear stress data for scenarios with and without devices present.
- Optional:
  - Receptor file: Contains critical shear stress values.
  - Probability/Boundary Condition file: Used to weight different run scenarios.

Output 
""""""
- GeoTIFF raster files: Visualize shear stress with and without devices, shear stress changes, and mobility classifications.

  - Output layers are interpolated onto structured grids.
  - **calculated_stressor.tif** : The probability weight difference between with devices and baseline models results. 

  - **calculated_stressor_with_receptor.tif**
    
    * Shear Stress: the mobility (Tau/TauCrit) difference using the grain size in the receptor file.

  - **receptor.tif** : the receptor file interpolated to the same grid as the output
  - **calculated_stressor_reclassified.tif** : 
    
    * Shear Stress: reclassified into increase erosion or deposition compared to the no device model run.

- CSV files: Contain statistics of area changes and mobility classifications.

  * The stressor values are binned into 25 bins and the surface area in which that change occurred, the percent of the overall model domain, and number of cells within the stressor is saved to a csv file.   
    + Lat/Lon converted to UTM (meter) coordinates for calculation.
    + UTM remains in the original unit of measure

- When a receptor is included, the stressor and stressor with receptor values are further segmented by unique receptor values.
  
  * For acoustics, the threshold exceeded, the species percent, and species density are generated.

- For Shear Stress and Velocity, the area of each unique reclassified value is defined and for each unique receptor value when included. 




Sources
"""""""

Default
+++++++

SEAT is designed to read Delft3D, DelftFM \*.map data files with structured and unstructured grids for shear stress and velocity.

- Shear Stress variables:

  * Structured : TAUMAX
  * Unstructured : taus

- Velocity variables:

  * Structured : U1, V1
  * Unstructured : ucxa, ucya

Coordinates are determined from the variable attributes

Alternative
+++++++++++

- Structured
 
  * Variable name TAUMAX
  * concatenated model runs [run number, depth, xcor, ycor]
  * Individual model runs [depth, xcor, ycor]

- Unstructured
  
  * Variable name taus
  * concatenated model runs [run number, depth, xcor, ycor]
  * Individual model runs [depth, xcor, ycor]

- Coordinate variable names must be in the variable attributes such that: 
  
  * xcor, ycor = netcdf_dataset.variables[variable].coordinates.split() 

Core Functions:
^^^^^^^^^^^^^^^

+--------------------------------------------+------------------------------------------------------------------+
| Function                                   | Description                                                      |
+============================================+==================================================================+
| ``critical_shear_stress()``                | Calculates critical shear stress from grain size.                |
+--------------------------------------------+------------------------------------------------------------------+
| ``classify_mobility()``                    | Classifies sediment mobility from device runs to no device runs. |
+--------------------------------------------+------------------------------------------------------------------+
| ``check_grid_define_vars()``               | Determines the type of grid and corresponding shear stress       |
|                                            | variable name and coordinate names.                              |
+--------------------------------------------+------------------------------------------------------------------+
| ``calculate_shear_stress_stressors()``     | Calculates the stressor layers as arrays from model and parameter|
|                                            | input.                                                           |
+--------------------------------------------+------------------------------------------------------------------+
| ``run_shear_stress_stressor()``            | Creates GeoTIFFs and area change statistics files for shear      |
|                                            | stress change.                                                   |
+--------------------------------------------+------------------------------------------------------------------+

