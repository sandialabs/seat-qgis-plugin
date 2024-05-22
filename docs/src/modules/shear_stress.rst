
Shear Stress Module
-------------------

The `shear_stress_module.py` is a component of SEAT. 
It assesses the impact of energy devices on shear stress and, consequently, sediment mobility in aquatic environment by comparing shear stress values with and without energy devices present.

Is there more theory to insert here? 

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

