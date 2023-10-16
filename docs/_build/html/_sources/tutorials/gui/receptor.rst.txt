Receptor (optional)
-------------------

The receptor file, which can be either a .csv or .tif, provides supplementary criteria to be input into each module.

Shear Stress Receptor:

- **File Type**: The receptor can be either a .csv file or a geotiff (\*.tif) file.

- **Content**: This file represents grain size, measured in microns (µm).

  - **Geotif Details**: If using a geotif file, it will be interpolated to align with the grid points of the model files, whether they are structured or unstructured. It is crucial that the geotif has the same projection and datum as the model files.

  - **CSV Details**: The CSV version of the receptor currently supports only a single grain size value and doesn't require any column headers. The file is structured simply, with the grain size value given directly.


Shear Stress
^^^^^^^^^^^^

.. figure:: ../../media/receptor_file_input.webp
   :scale: 100 %
   :alt: Receptor File

- A receptor file (.csv, or .tif) allows for additional criterion to be passed to each module. 

  - Shear stress : receptor is a \*.csv or geotif (\*.tif) of grain size in microns (µm). 
  
    * Geotif will be interpolated to the same grid points as the model files (structured or unstructured), must have same projection/datum.
    * The csv file currently only takes a single grain size and is formatted as below with no column headers.

.. figure:: ../../media/grain_size.webp
   :scale: 150 %
   :alt: Grain size

Velocity
^^^^^^^^

- Velocity: receptor is a (.csv) or geotif of critical velocity in units of meters per second (m/s).
  
  - Geotif will be interpolated to the same grid points as the model files (structured or unstructured), must have same projection/datum.
  - The csv file currently only takes a critical velocity and is formatted as below with no column headers.

.. figure:: ../../media/critical_velocity.webp
   :scale: 150 %
   :alt: Critical velocity

ParAcousti
^^^^^^^^^^

- ParAcousti: receptor is a \*.csv file with values indicating how to apply thresholds, grid scaling, and variables to use. 

  - species : optional but can be used for output
  - Paracousti Variable: Depending on the species, different variables might be needed and or different weightings. 
  - Threshold (db re 1 uPa): threshold above which negative impacts are expected. Units should match Paracousti Variable.
  - Depth Averaging (default DepthMax): 	

    * DepthMax: use the maximum value for each vertical column
    * DepthAverage: use the average value for each vertical column
    * Top: use the top/surface bin for each vertical column
    * Bottom: use the bottom/bed bin for each vertical column
  - species file averaged area (km2): the cumulative area over which each cell represents species percent and density (used to scale to each paracousti grid cell). Leave blank or set to 0 to prevent scaling. 

.. figure:: ../../media/paracousti_receptor.webp
   :scale: 100 %
   :alt: ParAcousti Receptor