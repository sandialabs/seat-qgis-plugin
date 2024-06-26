
Velocity Module
---------------

The SEAT velocity module evaluates the risk associated with changes in water velocity due to the presence of WECs and CECs by combining the velocity values with a receptor threshold layer. The receptor threshold layer is a map or single value for a critical velocity for species motility, such a larval motility. For the SEAT implementation, values above the threshold experience a change in motility. Output from this analysis includes the difference in the combined probability of velocity change over the range of modeled conditions (defined in the hydrodynamic probability file), difference in the estimated transport parameter, and estimate of the change in motility regime. 


Input 
^^^^^^^^
The input for the velocity module is a set of NetCDF files containing velocity data for scenarios with and without devices present and optional files for additional calculations. 

These include:

  - NetCDF files: model output with 1) baseline runs; 2) devices present runs.
  - \*Receptor file (GeoTIFF or CSV): Contains critical velocity values (e.g., a critical velocity for larval transport).
  - \*Model Probability Condition file (CSV): Used to weight different run scenarios.
  - \*Risk layer file (GeoTIFF): Contains spatial classifications, used to evaluate the impact of the devices on the environment.

  \* Optional input files


Input File Sources
"""""""""""""""""""""
By default, SEAT is designed to read output from the MHKit friendly tools (i.e., SNL-SWAN, SNL-Delft3D-CEC). 
The user can also provide inputs from other models, but the user must ensure that the input files are formatted correctly as described below.


Default Sources (MHkit friendly tools)
++++++++++++++++++++++++++++++++++++++++++
The MHkit friendly tools output shear stress data in NetCDF format from both structured and unstructured models.  
The NetCDF files contain the following variables:

**Structured**
  * ``U1``
  * ``V1``
**Unstructured** 
  * ``ucxa`` 
  * ``ucya``

Alternative Sources
+++++++++++++++++++++
If the user would like to use a different model, the user must ensure that the input files are formatted correctly within the NetCDF.
If the input includes multiple depths, the velocity values will be averaged over depth.

**Structured**
 
  * Variable names must be ``U1`` and ``V1``
  * concatenated model runs with dimensions ``[run number, depth, xcor, ycor]``
  * Individual model runs with dimensions ``[depth, xcor, ycor]`` for each model run

**Unstructured**
  
  * Variable name must be ``ucxa`` and ``ucya``
  * concatenated model runs with dimensions ``[run number, depth, xcor, ycor]``
  * Individual model runs with dimensions ``[depth, xcor, ycor]`` for each model run 

Coordinate variable names must be in the variable attributes such that: 
  * ``xcor, ycor = netcdf_dataset.variables[variable].coordinates.split()``


Output 
^^^^^^^^

GeoTIFF raster files
""""""""""""""""""""""
The output layers are interpolated onto structured grids in order to visualize velocity values with and without devices, velocity changes, and motility classifications.
Output is saved in the **Velocity Module** subdirectory. 

    - **velocity_magnitude_with_devices**: The probability weighted velocity with devices.
    - **velocity_magnitude_without_devices.tif**: The probability weighted velocity without devices.    
    - **velocity_magnitude_difference.tif** : The probability weight difference between with devices and baseline models results. 
    - **motility_with_devices**: The motility (Vel/VelCrit) with devices using the critical velocity receptor file.
    - **motility_without_devices**: The motility (Vel/VelCrit) without devices using the critical velocity receptor file.
    - **motility_difference**: The motility (Vel/VelCrit) difference between motility with devices and baseline models results using the critical velocity receptor file.
    - **critical_velocity.tif** : the receptor file interpolated to the same grid as the output
    - **motility_classified.tif** : reclassified into increased motility or decreased motility compared to the baseline model run.
    - **velocity_risk_layer.tif** :  the risk layer interpolated to the same grid as the output

The CSV files contain statistics of area calculations for various layers. If decimal degree coordinates are provided, the values are converted to UTM (meter) coordinates for calculations.
The stressor values are binned into 25 bins and associated with the surface area in which that change occurred, 
the percent of the overall model domain, and number of cells within the stressor is saved to a csv file.  

CSV files:    
      - **velocity_magnitude_difference.csv**
      - **motility_difference.csv**
      - **motility_classified.csv**

Segmented by velocity threshold when using the threshold receptor file:
 
      - **velocity_magnitude_difference_at_critical_velocity.csv**
      - **motility_difference_at_critical_velocity.csv**
      - **motility_classified_at_critical_velocity.csv**

Segmented by spatial classification when using the risk layer file: 

      - **velocity_magnitude_difference_at_velocity_risk_layer.csv**
      - **motility_difference_at_velocity_risk_layer.csv**
      - **motility_classified_at_velocity_risk_layer.csv**


Core Functions
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

