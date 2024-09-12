Acoustic Effects
^^^^^^^^^^^^^^^^

The acoustic effects from the WEC array at PacWave can be evaluated using the Acoustic module in the SEAT GUI. This module reads in individual ParAcousti model .nc files that correspond to wave conditions. 


Input
""""""

At this point you should have already setup the input files and style files as detailed in :ref:`prepare_tutorial_files`.

To run this demonstration, use the **Load GUI Inputs** button located at the bottom left of the SEAT GUI, navigate to :file:`DEMO_acoustics/pacwave/`, and there will be three files available to choose from:

    - :file:`acoustics_module_SEL_199db_threshold.ini`: Sound exposure level with a 199db threshold
    - :file:`acoustics_module_SEL_HFC_173db_threshold.ini`: Sound exposure level for high frequency creteceans with a 173db threshold
    - :file:`acoustics_module_SEL_LFC_199db_threshold.ini`: Sound exposure level for low frequency creteceans with a 199db threshold



Click on one, and click OK to load the inputs. If you need detailed instructions on how to load inputs, 
please refer to the :ref:`save_load_config` section in the :ref:`seat_qgis_plugin` documention.

.. Note::
   Your paths will differ than the ones shown in the example below. If you get an error double check the paths making sure the files exist at the specified location.

.. figure:: ../../media/pacwave_acoustics_input.webp
   :scale: 100 %
   :alt: Tanana sedimentation example input


Output
""""""""

For a given probability of occurrence of each wave condition, the combined annual acoustic effects can be estimated. SEAT generates a similar stressor layer consisting of the difference between the acoustic effects with and without the array. With a provided receptor file which consists of information regarding the species, threshold value, weighting, and variable used, a threshold map is generated as a percentage of time (based on the probability distribution) that a threshold will be exceeded. For demonstration purposes, an artificially low threshold is used to generate the percent exceeded threshold figure below.


.. list-table:: 
   :widths: 50 50
   :class: image-matrix

   * - .. image:: ../../media/PacWave_acoustics_stressor_layers.webp
         :scale: 125 %
         :alt: Layers
         :align: center

       .. raw:: html

          <div style="text-align: center;">Layers Legend</div>

     - .. image:: ../../media/PacWave_acoustics_risk_layer.webp
         :scale: 35 %
         :alt: Risk Layer
         :align: center

       .. raw:: html

          <div style="text-align: center;">Risk Layer</div>

   * - .. image:: ../../media/PacWave_acoustics_threshold_exceeded_receptor.webp
         :scale: 35 %
         :alt: Species Threshold Exceeded
         :align: center

       .. raw:: html

          <div style="text-align: center;">Species Threshold Exceeded</div>

     - .. image:: ../../media/PacWave_acoustics_calculated_paracousti.webp
         :scale: 35 %
         :alt: Calculated Paracousti
         :align: center

       .. raw:: html

          <div style="text-align: center;">Calculated Paracousti</div>

**Output Files**

Additional output files can be found in the specifed Output folder

.. code-block::

   Output
   └───ACOUSTICS_MODULE_100DB_THRESHOLD
       └───Acoustics Module
            paracousti_risk_layer.tif
            paracousti_stressor.csv
            paracousti_stressor.tif
            paracousti_stressor_at_paracousti_risk_layer.csv
            paracousti_without_devices.csv
            paracousti_without_devices.tif
            paracousti_with_devices.csv
            paracousti_with_devices.tif
            species_density.csv
            species_density.tif
            species_density_at_paracousti_risk_layer.csv
            species_percent.csv
            species_percent.tif
            species_percent_at_paracousti_risk_layer.csv
            species_threshold_exceeded.csv
            species_threshold_exceeded.tif
            species_threshold_exceeded_at_paracousti_risk_layer.csv