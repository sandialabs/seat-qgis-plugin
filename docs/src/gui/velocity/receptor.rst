Velocity Larval Motility Threshold (Optional)
------------------------------------------------

The receptor file serves as an additional input to each module, which can either be in .csv or .tif format. 

.. figure:: ../../media/receptor_file_input.webp
   :scale: 100 %
   :alt: Receptor File

Represents critical velocity, measured in meters per second (m/s).

- **File Type**: Supports .csv or geotiff (.tif) file formats.

  - **Geotiff Details**:
    
    - Interpolated to align with the model files' grid points (structured/unstructured).
    - Must have the same projection and datum as the model files.

  - **CSV Details**:
    
    - Supports only a single critical velocity value.
    - No column headers required; the file should contain the critical velocity value directly.
    - E.g.:

    +--------------------------+-------+
    | critical_velocity (m/s)  |  0.05 |
    +--------------------------+-------+