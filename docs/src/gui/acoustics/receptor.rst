Species Threshold File (Optional)
--------------------------------

The receptor file serves as an additional input to each module, which can either be in .csv or .tif format. 

.. figure:: ../../media/receptor_file_input.webp
   :scale: 100 %
   :alt: Receptor File



3. ParAcousti Receptor
^^^^^^^^^^^^^^^^^^^^^^^^

Contains values indicating thresholds, grid scaling, and variable selections. 

- **File Type**: Supports only .csv file format.

  - **Species**: Optional; can be used for output.
  - **ParAcousti Variable**: Variables and weightings may vary depending on species.
  - **Threshold (db re 1 uPa)**: Threshold above which negative impacts are expected. Units should match Paracousti Variable.
  - **Depth Averaging** (Default: DepthMax):

    - DepthMax: Use the maximum value for each vertical column.
    - DepthAverage: Use the average value for each vertical column.
    - Top: Use the top/surface bin for each vertical column.
    - Bottom: Use the bottom/bed bin for each vertical column.
  
  - **Species File Averaged Area (km²)**: Represents cumulative area for each cell regarding species percent and density; used for scaling to each ParAcousti grid cell. Leave blank or set to 0 to prevent scaling. 
  - E.g.:

  +----------------------------------+------------+
  | species                          | Blue Whale |
  +----------------------------------+------------+
  | Paracousti Variable              | totSPL     |
  +----------------------------------+------------+
  | Threshold (dB re 1uPa)           | 219        |
  +----------------------------------+------------+
  | Depth Averaging                  | DepthMax   |
  +----------------------------------+------------+
  | species file averaged area (km2) | 625        |
  +----------------------------------+------------+

