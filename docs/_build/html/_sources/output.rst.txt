.. _output:

Output maps
===========

Output Files
------------

- Output layers are interpolated onto structured grids.
- **calculated_stressor.tif** : The probability weight difference between with devices and baseline models results. 

  * for acoustics assumes baseline=0 if no baseline model files provided.

- **calculated_stressor_with_receptor.tif**
  
  * Shear Stress: the mobility (Tau/TauCrit) difference using the grain size in the receptor file.
  * Velocity: the motility (Vel/VelCrit) difference using the critical velocity in the receptor file.

- **receptor.tif** : the receptor file interpolated to the same grid as the output
- **calculated_stressor_reclassified.tif** : 
  
  * Shear Stress: reclassified into increase erosion or deposition compared to the no device model run.
  * Velocity : reclassified into increase motility or no change compared to the no device model run.

- **calculate_paracousti.tif** : the calculated with device probability weighted paracousti file.
- **Threshold_exceeded_receptor.tif** : the percent of time the acoustic threshold was exceeded.
- **species_percent.tif** : the threshold exceeded and weighted species percent.
- **species_density.tif** : the threshold exceeded and weighted species density.

Stressor Area Change
--------------------

- The stressor values are binned into 25 bins and the surface area in which that change occurred, the percent of the overall model domain, and number of cells within the stressor is saved to a csv file. 
  
  * Lat/Lon converted to UTM (meter) coordinates for calculation.
  * UTM remains in the original unit of measure

- When a receptor is included, the stressor and stressor with receptor values are further segmented by unique receptor values.
  
  * For acoustics, the threshold exceeded, the species percent, and species density are generated.

- For Shear Stress and Velocity, the area of each unique reclassified value is defined and for each unique receptor value when included. 

Power Module
------------

When a directory is specified for the device power the following are generated.

- CSV:

  * BC_probability_wPower.csv : probabilities input file with appended power generated for each scenario
  * Obstacle_Matching.csv : Obstacle pairs corresponding to a single device and centroid X,Y.
  * Power_per_device_annual.csv : Total power generated (Watts) per device over the annual timespan (probabilities file).
  * Power_per_device_per_scenario.csv : Table of total power generated (Watts) with device (row), and power file (column).

- PNG:

  * Scaled_Power_per_device_per_scenario.webp : subplots of bar graph of power generated for each run per device.
  * Scaled_Power_per_device_per_obstacle.webp : subplots of bar graph of power generated for each run per obstacle.
  * Total_Scaled_Power_Bars_per_Run.webp : Bar graph of total power generated for each run scenario (probabilities file).
  * Total_Scaled_Power_Bars_per_obstacle.webp : Bar graph of total power generated for each obstacle.
  * Total_Scaled_Power_per_Device.webp : Bar graph of total power generated for each device
  * Obstacle_Locations.webp : Spatial plot of XY coordinates for each obstacle endpoint.
  * Device Number Locations.webp : Spatial plot of XY coordinates for each device.
  * Device_Power.webp : Spatial heat map of total power generated (mega watts) for each device.
