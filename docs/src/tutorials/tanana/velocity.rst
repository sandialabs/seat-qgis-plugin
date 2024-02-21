Larval Transport Analysis (Velocity)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This set of inputs evaluates the impact on larval motility given a single critical velocity receptor in a CSV file.

Output
""""""
  
For this case the velocity with devices is compared to the velocity without devices and a difference (stressor) is calculated.


.. list-table:: 
   :widths: 50 50
   :class: image-matrix

   * - .. image:: ../../media/tanana_velocity_layers.webp
         :scale: 70 %
         :alt: Layers
         :align: center

       .. raw:: html

          <div style="text-align: center;">Layers Legend</div>

     - .. image:: ../../media/tanana_velocity_stressor_reclassified.webp
         :scale: 25 %
         :alt: Calculated Stressor Reclassified
         :align: center

       .. raw:: html

          <div style="text-align: center;">Calculated Stressor Reclassified</div>

   * - .. image:: ../../media/tanana_velocity_stressor_with_receptor.webp
         :scale: 25 %
         :alt: Stressor with Receptor
         :align: center

       .. raw:: html

          <div style="text-align: center;">Stressor with Receptor</div>

     - .. image:: ../../media/tanana_velocity_stressor.webp
         :scale: 25 %
         :alt: Calculated Stressor
         :align: center

       .. raw:: html

          <div style="text-align: center;">Calculated Stressor</div>

**Output Files**

Additional output files can be found in the specifed Output folder.

.. code-block::

   Output
   └───Shear_and_Velocity
       └───Velocity Module
            motility_classified.csv
            motility_classified_at_critical_velocity.csv
            motility_difference.csv
            motility_difference_at_critical_velocity.csv
            velocity_magnitude_difference.csv
            velocity_magnitude_difference_at_critical_velocity.csv
            critical_velocity.tif
            motility_classified.tif
            motility_difference.tif
            motility_with_devices.tif
            motility_without_devices.tif
            velocity_magnitude_difference.tif
            velocity_magnitude_with_devices.tif
            velocity_magnitude_without_devices.tif