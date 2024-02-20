
Shear Stress Module
-------------------

The `shear_stress_module.py` is a component of SEAT. It's aimed at assessing the impact of shear stressors on sediment mobility in aquatic environments. This module provides insights into how various devices or conditions can affect shear stress by allowing comparison between scenarios with and without these elements present.


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
- Output is saved in the **Shear Stress** subdirectory.

  - Output layers are interpolated onto structured grids.
  - **shear_stress_with_devices.tif**: The probability weighted shear stress with devides.
  - **shear_stress_without_devices.tif**: The probability weighted shear stress without devices.
  - **shear_stress_difference.tif** : The probability weight difference between shear stress with devices and baseline models results. 
  - **sediment_mobility_with_devices**: The mobility (Tau/TauCrit) with devices using the grain size in the receptor file.
  - **sediment_mobility_without_devices**: The mobility (Tau/TauCrit) without devices using the grain size in the receptor file.
  - **sediment_mobility_difference**: The mobility (Tau/TauCrit) difference between shear stress with devices and baseline models results using the grain size in the receptor file.
  - **sediment_grain_size.tif** : the receptor file interpolated to the same grid as the output
  - **sediment_mobility_classified.tif** : reclassified into increased erosion or deposition compared to the baseline model run.
  - **shear_stress_risk_layer.tif** :  the risk layer interpolated to the same grid as the output
  - **shear_stress_risk_metric.tif** : A quantified risk metric based on Jones et al. (2018) Equation 7. <https://doi.org/10.3390/en11082036>

- CSV files: Contain statistics of area changes and mobility classifications.

  * The stressor values are binned into 25 bins and the surface area in which that change occurred, the percent of the overall model domain, and number of cells within the stressor is saved to a csv file. 
    - Output includes:
        - **shear_stress_difference.csv**
        - **sediment_mobility_difference.csv**
        - **sediment_mobility_classified.csv**
        - **shear_stress_risk_metric.csv**

    - When a grain size receptor is included, the values are further segmented by unique grain size values.
    - Output includes:
        - **shear_stress_difference_at_sediment_grain_size.csv**
        - **sediment_mobility_difference_at_sediment_grain_size.csv**
        - **sediment_mobility_classified_at_sediment_grain_size.csv**
        - **shear_stress_risk_metric_at_sediment_grain_size**

    - When a risk layer receptor is included, the values are further segmented by unique risk layer values.
    - Output includes:
        - **sediment_mobility_difference_at_shear_stress_risk_layer.csv**
        - **shear_stress_risk_metric_at_shear_stress_risk_layer.csv**

    + Lat/Lon converted to UTM (meter) coordinates for calculation.
    + UTM remains in the original unit of measure


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

