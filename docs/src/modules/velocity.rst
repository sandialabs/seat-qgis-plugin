
Velocity Module
---------------

The `velocity_module.py` is a component of SEAT aimed at assessing the impact of velocity stressors on larval motility in aquatic environments. This module provides insights into how various devices or conditions can affect velocity by allowing comparison between scenarios with and without these elements present.

Data
^^^^

Input 
""""""
- NetCDF files: Contain velocity data for scenarios with and without devices present.
- Optional:

  * Receptor file: Contains critical velocity values.
  * Probability/Boundary Condition file: Used to weight different run scenarios.

Output 
""""""
- GeoTIFF raster files: Visualize velocity with and without devices, velocity changes, and motility classifications.
- Output is saved in the **Velocity Module** subdirectory.

    - Output layers are interpolated onto structured grids.
    - **velocity_magnitude_with_devices**: The probability weighted velocity with devides.
    - **velocity_magnitude_without_devices.tif**: The probability weighted velocity without devices.    
    - **velocity_magnitude_difference.tif** : The probability weight difference between with devices and baseline models results. 
    - **motility_with_devices**: The motility (Vel/VelCrit) with devices using the critical velocity receptor file.
    - **motility_without_devices**: The motility (Vel/VelCrit) without devices using the critical velocity receptor file.
    - **motility_difference**: The motility (Vel/VelCrit) difference between motility with devices and baseline models results  using the critical velocity receptor file.
    - **critical_velocity.tif** : the receptor file interpolated to the same grid as the output
    - **motility_classified.tif** : reclassified into increased motility or decreased motility compared to the baseline model run.
    - **velocity_risk_layer.tif** :  the risk layer interpolated to the same grid as the output

- CSV files: Contain statistics of area changes and motility classifications.

  * The stressor values are binned into 25 bins and the surface area in which that change occurred, the percent of the overall model domain, and number of cells within the stressor is saved to a csv file. 
    
    - Output includes:
     
      - **velocity_magnitude_difference.csv**
      - **motility_difference.csv**
      - **motility_classified.csv**

    - When a critical velocity receptor is included, the values are further segmented by unique grain size values.
    - Output includes:
    
      - **velocity_magnitude_difference_at_critical_velocity.csv**
      - **motility_difference_at_critical_velocity.csv**
      - **motility_classified_at_critical_velocity.csv**

    - When a risk layer receptor is included, the values are further segmented by unique risk layer values.
    - Output includes:
    
      - **velocity_magnitude_difference_at_velocity_risk_layer.csv**
      - **motility_difference_at_velocity_risk_layer.csv**
      - **motility_classified_at_velocity_risk_layer.csv**

    + Lat/Lon converted to UTM (meter) coordinates for calculation.
    + UTM remains in the original unit of measure

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

- Structured

  * Variable names : U1, V1
  * concatenated model runs [run number, time, depth, xcor, ycor]
  * Individual model runs [time, depth, xcor, ycor]

- Unstructured

  * Variable names : ucxa, ucya
  * concatenated model runs [run number, time, depth, xcor, ycor]
  * Individual model runs [time, depth, xcor, ycor]

- Coordinate variable names must be in the variable attributes such that: 

  * xcor, ycor = netcdf_dataset.variables[variable].coordinates.split()


Core Functions:
^^^^^^^^^^^^^^^

+------------------------------------+-----------------------------------------------------------------------+
| Function                           | Description                                                           |
+====================================+=======================================================================+
| ``classify_motility()``            | This function classifies larval motility into various categories such |
|                                    | as Reduced, Increased, or New Motility based on the comparison of     |
|                                    | device runs and baseline (no device) runs.                            |
+------------------------------------+-----------------------------------------------------------------------+
| ``check_grid_define_vars()``       | Determines the type of grid (structured/unstructured) and defines     |
|                                    | corresponding velocity and coordinate variable names.                 |
+------------------------------------+-----------------------------------------------------------------------+
| ``calculate_velocity_stressors()`` | Main function that loads data, performs calculations, and computes    |
|                                    | various metrics including velocity differences and motility           |
|                                    | classifications.                                                      |
+------------------------------------+-----------------------------------------------------------------------+
| ``run_velocity_stressor()``        | Creates GeoTIFFs and CSV files to visualize and quantify velocity     |
|                                    | changes and motility classifications.                                 |
+------------------------------------+-----------------------------------------------------------------------+

