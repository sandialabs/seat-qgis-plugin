Output: 
------------
The output files come in two formats: 
    1. GeoTIFF raster files to visualize differences in shear stress 
    2. CSV files

Output is saved in: ``Output\Shear_and_Velocity\Shear Stress Module\`` directory

GeoTIFF raster files:
^^^^^^^^^^^^^^^^^^^^^
Output layers are interpolated onto structured grids and saved as geotiffs.

  
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
^^^^^^^^^^
+ Lat/Lon converted to UTM (meter) coordinates for calculation.
+ UTM remains in the original unit of measure

There are up to three sets of files depending on whether or not the grain size receptor and risk layer receptor are included. 

Baseline:
""""""""
Contain statistics of area changes and mobility classifications. Specifically, 
the stressor values are binned into 25 bins and the surface area in which that change occurred, 
the percent of the overall model domain, and number of cells within the stressor is saved to a csv file. 

    
    - **shear_stress_difference.csv**
    - **sediment_mobility_difference.csv**
    - **sediment_mobility_classified.csv**
    - **shear_stress_risk_metric.csv**

With Grain Receptor:
""""""""""""""""""""""
When a grain size receptor is included, the values are further segmented by unique grain size values.
    
      - **shear_stress_difference_at_sediment_grain_size.csv**
      - **sediment_mobility_difference_at_sediment_grain_size.csv**
      - **sediment_mobility_classified_at_sediment_grain_size.csv**
      - **shear_stress_risk_metric_at_sediment_grain_size**

With Risk Layer Receptor:
""""""""""""""""""""""
When a risk layer receptor is included, the values are further segmented by unique risk layer values.
    
      - **sediment_mobility_difference_at_shear_stress_risk_layer.csv**
      - **shear_stress_risk_metric_at_shear_stress_risk_layer.csv**

