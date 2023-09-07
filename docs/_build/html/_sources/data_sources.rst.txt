.. _data_sources:

Data Sources
============

Default Data Sources
--------------------

Shear Stress and Velocity Modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SEAT is designed to read Delft3D, DelftFM \*.map data files with structured and unstructured grids for shear stress and velocity.

- Shear Stress variables:

  * Structured : TAUMAX
  * Unstructured : taus

- Velocity variables:

  * Structured : U1, V1
  * Unstructured : ucxa, ucya

Coordinates are determined from the variable attributes

Acoustics Module
^^^^^^^^^^^^^^^^
The acoustics module is designed for paracousti data sources (https://sandialabs.github.io/Paracousti/). 

Acoustics variables:

- The variable is specified in the receptor file to allow for various weighting, sound pressure level, or sound exposure level thresholds. 

Coordinates are determined from the variable attributes


Alternative Data Sources
------------------------

Alternative datasets such as FVCOM, ROMS, etc. can be used by modifying the variable names in the dataset and ensuring data is in a netcdf format with appropriate dimensions.

Shear Stress
^^^^^^^^^^^^

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

Velocity
^^^^^^^^

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

Acoustics
^^^^^^^^^

The Acoustics module can utilize alternate datasets with the following requirements:

- Variable name must be specified in the receptor file.
- Variable attributes must include the coordinates variable names, such that:

  * xcor, ycor = netcdf_dataset.variables[variable].coordinates.split() 

- The coordinates units attribute must include “degrees” if the coordinates are lat/lon such that:

  * 'degrees’ in ds.variables[<xcor variable>].units is True for lat/lon