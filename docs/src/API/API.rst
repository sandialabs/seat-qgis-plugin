.. _API:

API
====

+--------------------------------------------+------------------------------------------------------------------+
| Function                                   | Description                                                      |
+============================================+==================================================================+
| ``create_species_array()``                 | Interpolates or creates an array of percent or density of species|
|                                            | from input files and coordinates.                                |
+--------------------------------------------+------------------------------------------------------------------+
| ``calculate_acoustic_stressors()``         | Calculates the stressor layers as arrays from model and parameter|
|                                            | input.                                                           |
+--------------------------------------------+------------------------------------------------------------------+
| ``run_acoustics_stressor()``               | Creates GeoTIFFs and area change statistics files for acoustic   |
|                                            | stressor change.                                                 |
+--------------------------------------------+------------------------------------------------------------------+
| ``redefine_structured_grid()``             | (From `stressor_utils`) Redefines grids to regular spacing, used |
|                                            | in `calculate_acoustic_stressors`.                               |
+--------------------------------------------+------------------------------------------------------------------+
| ``resample_structured_grid()``             | (From `stressor_utils`) Resamples grids, used in                 |
|                                            | `calculate_acoustic_stressors`.                                  |
+--------------------------------------------+------------------------------------------------------------------+

