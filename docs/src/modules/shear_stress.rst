
Shear Stress Module
-------------------
The SEAT shear stress module evaluates the impacts of sediment mobilization over defined risk layers for the installation of Marine Energy Converters (Wave Energy Converter (WEC) and Current Energy Converter (CEC)) arrays for either wave or current flow conditions. The input is numerical model simulation from SNL-SWAN, SNL-Delft3d-CEC, or any similarly formatted NetCDF file and an optional hydrodynamic probabilities file. The threshold receptor for this analysis is a grain size map from which a critical shear stress for sediment mobility is calculated. Using the ratio between the calculated shear stress and the critical shear stress for sediment mobility, a transport parameter is calculated for both the baseline and MEC simulations. The relative change between these two values defines the mobilization and depositional regime (i.e. erosional and accretional) and whether the regime changes with the presence of MECs. A risk metric, defined in Jones et al (2018), classifies the type of changes associated with changes in sediment mobility. Output from this analysis include the summaries of the difference in shear stress across modeled conditions with and without devices, the resulting difference in the transport potential based on bed sediment properties, and derivation of a risk metric that represents the potential for environmental change. SEAT can also evaluate the impact to various user defined benthic characteristics, such as vegetation, marine species habitat, sediment characteristics, marine infrastructure (shipping channels, buried cables, etc.), or contaminated sediments with the inclusion of an additional risk layer.

Input 
^^^^^^
The input for the shear stress module is a set of NetCDF files containing shear stress data for scenarios with and without devices present and optional files for additional calculations. 
These include:

  - NetCDF files: model output with 1) baseline runs; 2) devices present runs.
  - \*Receptor file (GeoTIFF or CSV): Contains grain size values.
  - \*Model Probability Condition file (CSV): Contains model weights, used to weight different run scenarios.
  - \*Risk layer file (GeoTIFF): contains spatial classifications, used to evaluate the impact of the devices on the environment.

  \* Optional input files

Input File Sources
""""""""""""""""""""""
By default, SEAT is designed to read output from the MHKit friendly tools (i.e., SNL-SWAN, SNL-Delft3D-CEC). 
The user can also provide inputs from other models, but the user must ensure that the input files are formatted correctly as described below.

Default Sources (MHkit friendly tools)
++++++++++++++++++++++++++++++++++++++++++
The MHkit friendly tools output shear stress data in NetCDF format from both structured and unstructured models. 
The NetCDF files contain the following variables:

**Structured**
  * ``TAUMAX``
**Unstructured** 
  * ``taus``


Alternative Sources
+++++++++++++++++++++
If the user would like to use a different model, the user must ensure that the input files are formatted correctly within the NetCDF.

**Structured**
 
  * Variable name must be ``TAUMAX``
  * concatenated model runs with dimensions ``[run number, depth, xcor, ycor]``
  * Individual model runs with dimensions ``[depth, xcor, ycor]`` for each model run

**Unstructured**
  
  * Variable name must be ``taus``
  * concatenated model runs with dimensions ``[run number, depth, xcor, ycor]``
  * Individual model runs with dimensions ``[depth, xcor, ycor]`` for each model run 

Coordinate variable names must be in the variable attributes such that ``xcor, ycor = netcdf_dataset.variables[variable].coordinates.split()``


Output 
^^^^^^^

GeoTIFF raster files
""""""""""""""""""""""
The output layers are interpolated onto structured grids in order to visualize shear stress with and without devices, shear stress changes, and mobility classifications.
Output is saved in the **Shear Stress** subdirectory. 

  - **shear_stress_with_devices.tif**: The probability weighted shear stress with devices.
  - **shear_stress_without_devices.tif**: The probability weighted shear stress without devices.
  - **shear_stress_difference.tif** : The probability weight difference between shear stress with devices and baseline models results. 
  - **sediment_mobility_with_devices**: The mobility (Tau/TauCrit) with devices using the grain size in the receptor file.
  - **sediment_mobility_without_devices**: The mobility (Tau/TauCrit) without devices using the grain size in the receptor file.
  - **sediment_mobility_difference**: The mobility (Tau/TauCrit) difference between shear stress with devices and baseline models results using the grain size in the receptor file.
  - **sediment_grain_size.tif** : the receptor file interpolated to the same grid as the output
  - **sediment_mobility_classified.tif** : reclassified into increased erosion or deposition compared to the baseline model run.
  - **shear_stress_risk_layer.tif** :  the risk layer interpolated to the same grid as the output
  - **shear_stress_risk_metric.tif** : A quantified risk metric based on `Jones et al. (2018) Equation 7 <https://doi.org/10.3390/en11082036>`_

CSV files
""""""""""""
These files contain statistics of shear stress changes in different areas. In these files, latitude and longitude are converted to UTM coordinates and meters from the UTM grid are used as measurements.
The stressor values are binned into 25 bins and the surface area in which that change occurred, 
the percent of the overall model domain, and number of cells within the stressor is saved to a csv file. 

      - **shear_stress_difference.csv**
      - **sediment_mobility_difference.csv**
      - **sediment_mobility_classified.csv**
      - **shear_stress_risk_metric.csv**

Segmented by grain size when using the grain size receptor file:
    
      - **shear_stress_difference_at_sediment_grain_size.csv**
      - **sediment_mobility_difference_at_sediment_grain_size.csv**
      - **sediment_mobility_classified_at_sediment_grain_size.csv**
      - **shear_stress_risk_metric_at_sediment_grain_size**

Segmented by spatial classification when using the risk layer file: 

      - **sediment_mobility_difference_at_shear_stress_risk_layer.csv**
      - **shear_stress_risk_metric_at_shear_stress_risk_layer.csv**



Core Functions
^^^^^^^^^^^^^^^
The shear stress module contains the following core functions:

.. list-table:: Core Functions
  :widths: 25 75
  :header-rows: 1

  * - Function
    - Description
  * - ``critical_shear_stress()``
    - Calculates critical shear stress from grain size.
  * - ``classify_mobility()``
    - Classifies sediment mobility from device runs to no device runs.
  * - ``check_grid_define_vars()``
    - Determines the type of grid and corresponding shear stress variable name and coordinate names.
  * - ``calculate_shear_stress_stressors()``
    - Calculates the stressor layers as arrays from model and parameter input.
  * - ``run_shear_stress_stressor()``
    - Creates GeoTIFFs and area change statistics files for shear stress change.