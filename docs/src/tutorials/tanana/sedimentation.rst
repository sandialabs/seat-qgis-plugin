Sedimentation Analysis (Shear Stress)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evaluate the impact of CEC devices on sediment mobility considering a single median grain size receptor in a CSV file.

Output
""""""

The above input set evaluates the impact on sediment mobility considering a single median grain size receptor in a CSV file. The probability-weighted shear stress with devices is compared to the scenario without devices. The resulting stressor (change in shear stress), stressor with receptor (change in sediment mobility), and the reclassified stressor from the analysis are illustrated below.

Each layer will look as shown below. To add the map layer see the :ref:`map` section. 

.. list-table:: 
   :widths: 50 50
   :class: image-matrix

   * - .. image:: ../../media/tanana_shear_stress_layers.webp
         :scale: 70 %
         :alt: Layers
         :align: center

       .. raw:: html

          <div style="text-align: center;">Layers Legend</div>

     - .. image:: ../../media/tanana_shear_stress_stressor_reclassified.webp
         :scale: 25 %
         :alt: Calculated Stressor Reclassified
         :align: center

       .. raw:: html

          <div style="text-align: center;">Calculated Stressor Reclassified</div>

   * - .. image:: ../../media/tanana_shear_stress_stressor_with_receptor.webp
         :scale: 25 %
         :alt: Stressor with Receptor
         :align: center

       .. raw:: html

          <div style="text-align: center;">Stressor with Receptor</div>

     - .. image:: ../../media/tanana_shear_stress_stressor.webp
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
       └───Shear Stress module
            sediment_mobility_classified.csv
            sediment_mobility_classified_at_sediment_grain_size.csv
            sediment_mobility_difference.csv
            sediment_mobility_difference_at_sediment_grain_size.csv
            shear_stress_difference.csv
            sediment_mobility_difference_at_sediment_grain_size.csv
            shear_stress_difference.csv
            shear_stress_difference_at_sediment_grain_size.csv
            shear_stress_risk_metric.csv
            shear_stress_risk_metric_at_sediment_grain_size.csv
            sediment_grain_size.tif
            sediment_mobility_classified.tif
            sediment_mobility_difference.tif
            sediment_mobility_with_devices.tif
            sediment_mobility_without_devices.tif
            shear_stress_difference.tif
            shear_stress_risk_metric.tif
            shear_stress_with_devices.tif
            shear_stress_without_devices.tif