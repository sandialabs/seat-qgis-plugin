
Power Module
------------

The `power_module.py` is a component of SEAT crafted to calculate the power output from a device array, providing a systematic means of evaluating and visualizing the spatial distribution, location, and power output from the devices in a specific marine environment.

Data
^^^^
Input 
""""""
- .OUT files: Contain power data for different scenarios.
- .pol file: Contains information on the obstacle polygon configurations.
- Optional:
  - Probability/Boundary Condition file: Used to weight different run scenarios.

Output 
""""""
- CSV files: Contain power data, obstacle matching data, and annual power per device.
- PNG files: Visualizations of obstacle locations, device number locations, total scaled power bars per run, scaled power per device per scenario, total scaled power per device, and device power heatmap.

Core Functions:
^^^^^^^^^^^^^^^

+--------------------------------------------+------------------------------------------------------------------+
| Function                                   | Description                                                      |
+============================================+==================================================================+
| ``read_obstacle_polygon_file()``           | Reads the obstacle polygon file to obtain xy coordinates of each |
|                                            | obstacle.                                                        |
+--------------------------------------------+------------------------------------------------------------------+
| ``find_mean_point_of_obstacle_polygon()``  | Calculates the center of each obstacle based on xy coordinates.  |
+--------------------------------------------+------------------------------------------------------------------+
| ``plot_test_obstacle_locations()``         | Creates a plot showing the spatial distribution and location of  |
|                                            | each obstacle.                                                   |
+--------------------------------------------+------------------------------------------------------------------+
| ``centroid_diffs()``                       | Determines the closest centroid pair among obstacles.            |
+--------------------------------------------+------------------------------------------------------------------+
| ``extract_device_location()``              | Creates a dictionary summary of each device location.            |
+--------------------------------------------+------------------------------------------------------------------+
| ``pair_devices()``                         | Determines the two intersecting obstacles that create a device.  |
+--------------------------------------------+------------------------------------------------------------------+
| ``create_power_heatmap()``                 | Creates a heatmap visualizing device location and power output.  |
+--------------------------------------------+------------------------------------------------------------------+
| ``read_power_file()``                      | Reads power file and extracts final set of converged data.       |
+--------------------------------------------+------------------------------------------------------------------+
| ``sort_data_files_by_runorder()``          | Sorts data files by run order based on boundary conditions data. |
+--------------------------------------------+------------------------------------------------------------------+
| ``sort_bc_data_by_runorder()``             | Sorts boundary condition data by run order.                      |
+--------------------------------------------+------------------------------------------------------------------+
| ``reset_bc_data_order()``                  | Resets the order of boundary condition data.                     |
+--------------------------------------------+------------------------------------------------------------------+
| ``calculate_power()``                      | Reads the power files, calculates the total annual power based   |
|                                            | on hydrodynamic probabilities, and saves data and visualizations.|
+--------------------------------------------+------------------------------------------------------------------+
