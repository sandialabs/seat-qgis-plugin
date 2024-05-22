Shear Stress
=====================

**Shear Stress Module**: Analyzes the spatial changes in bed mobility. 

The shear stress module takes 3 required inputs: 
   1. Baseline Model Results
   2. Device Model Results
   3. Model Probabilities
   4. Temporal Averaging

The following are optional inputs:
   5. Bed Sediment Grain Size
   6. Risk Layer


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


.. figure:: ../../media/SEAT_GUI.webp
   :scale: 90 %
   :alt: SEAT's main GUI window


.. toctree::
   :maxdepth: 1

   01_model_results_dir
   02_probabilities.rst
   03_bed_sediment.rst
   04_risk_layer.rst
   05_temporal_avg.rst