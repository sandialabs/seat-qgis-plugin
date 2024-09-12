Risk layer (Optional)
--------------------------------

The risk layer is a receptor file that serves as an additional input to each module and designated which layers are sensitive and would be effected by the acoustics. 
It must be a numerically classified .tif format. It is the same as what is used in the shear stress and velocity modules.

.. figure:: ../../media/risk_layer_gui_input.webp
   :scale: 100 %
   :alt: Risk Layer File

Represents a layer to evalute change against. Examples include vegetation habitat, marine ecosystems, contaminated sediments, marine protected areas, or archaeological artifacts.

- **File Type**: Supports geotiff (.tif) file format.
  
  - **Geotiff Details**:

    - Must have the same projection and datum as the model files.
    - Will be nearest-neighbor interpolated to align with the model files' grid points (structured/unstructured).
    - Must be integer classified eg. (0 = 'Kelp', 1 = 'Rock')