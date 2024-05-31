
Shear Stress Module
-------------------

The shear stress module determines how instruments might change the shear stress and sediment mobility. 

Input (how is this different from Sources?)
^^^^^^
- NetCDF files: Contain shear stress data for scenarios with and without devices present.
- Optional:

  - Receptor file: Contains critical shear stress values.
  - Probability/Boundary Condition file: Used to weight different run scenarios.

Output 
^^^^^^^

GeoTIFF raster files
""""""""""""""""""""""
Visualize shear stress with and without devices, shear stress changes, and mobility classifications.
Output is saved in the **Shear Stress** subdirectory. The output layers are interpolated onto structured grids.

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

CSV files
""""""""""""
These files contain statistics of shear stress changes in different areas. In these files, latitude and longitude are converted to UTM coordinates and meters from the UTM grid are used as measurements.
The stressor values are binned into 25 bins and the surface area in which that change occurred, 
the percent of the overall model domain, and number of cells within the stressor is saved to a csv file. 

      - **shear_stress_difference.csv**
      - **sediment_mobility_difference.csv**
      - **sediment_mobility_classified.csv**
      - **shear_stress_risk_metric.csv**

Segmented by grain size:
    
      - **shear_stress_difference_at_sediment_grain_size.csv**
      - **sediment_mobility_difference_at_sediment_grain_size.csv**
      - **sediment_mobility_classified_at_sediment_grain_size.csv**
      - **shear_stress_risk_metric_at_sediment_grain_size**

Segmented with respect to risk layer: 

      - **sediment_mobility_difference_at_shear_stress_risk_layer.csv**
      - **shear_stress_risk_metric_at_shear_stress_risk_layer.csv**



Sources
"""""""

Default
+++++++

SEAT is designed to read shear stress and velocity variables from Delft3D, DelftFM \*.map data files for either structured or unstructured grids. The coordinates are determined from the variable attributes


- Shear Stress variables:

  * Structured : TAUMAX
  * Unstructured : taus

- Velocity variables:

  * Structured : U1, V1
  * Unstructured : ucxa, ucya



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

