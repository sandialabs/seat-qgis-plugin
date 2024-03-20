Acoustic Effects
^^^^^^^^^^^^^^^^

The acoustic effects from the WEC array at PacWave can be evaluated using the Acoustic module in the SEAT GUI. This module reads in individual ParAcousti model .nc files that correspond to wave conditions. 


.. To run this demonstration, use the **Load GUI Inputs** button located at the bottom left of the SEAT GUI, navigate to :file:`DEMO/DEMO paracousti/demo_paracousti_with_receptor_219.ini`, and click OK to load the inputs. If you need detailed instructions on how to load inputs, please refer to the :ref:`save_load_config` section in the :ref:`gui` documention.


Output
""""""""

For a given probability of occurrence of each wave condition, the combined annual acoustic effects can be estimated. SEAT generates a similar stressor layer consisting of the difference between the acoustic effects with and without the array. With a provided receptor file which consists of information regarding the species, threshold value, weighting, and variable used, a threshold map is generated as a percentage of time (based on the probability distribution) that a threshold will be exceeded. For demonstration purposes, an artificially low threshold is used to generate the percent exceeded threshold figure below.


.. list-table:: 
   :widths: 50 50
   :class: image-matrix

   * - .. image:: ../../media/PacWave_acoustics_stressor_layers.webp
         :scale: 70 %
         :alt: Layers
         :align: center

       .. raw:: html

          <div style="text-align: center;">Layers Legend</div>

     - .. image:: ../../media/PacWave_acoustics_calculated_stressor.webp
         :scale: 25 %
         :alt: Calculated Stressor
         :align: center

       .. raw:: html

          <div style="text-align: center;">Calculated Stressor</div>

   * - .. image:: ../../media/PacWave_acoustics_calculated_paracousti.webp
         :scale: 25 %
         :alt: Calculated Paracousti
         :align: center

       .. raw:: html

          <div style="text-align: center;">Calculated Paracousti</div>

     - .. image:: ../../media/PacWave_acoustics_threshold_exceeded_receptor.webp
         :scale: 25 %
         :alt: Threshold Exceeded Receptor
         :align: center

       .. raw:: html

          <div style="text-align: center;">Threshold Exceeded Receptor</div>

**Output Files**

Additional output files can be found in the specifed Output folder

.. code-block::

   Output
   └───All_Modules
       └───Acoustics Module
            paracousti_stressor.csv
            paracousti_stressor_at_paracousti_risk_layer.csv
            paracousti_with_devices.csv
            paracousti_without_devices.csv
            species_density.csv
            species_density_at_paracousti_risk_layer.csv
            species_percent.csv
            species_percent_at_paracousti_risk_layer.csv
            species_threshold_exceeded.csv
            species_threshold_exceeded_at_paracousti_risk_layer.csv
            paracousti_risk_layer.tif
            paracousti_stressor.tif
            paracousti_with_devices.tif
            paracousti_without_devices.tif
            species_density.tif
            species_percent.tif
            species_threshold_exceeded.tif