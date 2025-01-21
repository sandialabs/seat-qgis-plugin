
PacWave Site
------------

.. toctree::
   :maxdepth: 1
   :hidden:

   shear_stress.rst
   velocity.rst
   acoustics.rst
   power.rst

PacWave is a designated area for marine energy testing off the coast of Oregon. This site has been the focus of model development for SEAT application. A coupled hydrodynamic and wave model was developed using SNL-SWAN and Delft3D-Flow. A range of site conditions is listed in the Model Probabilities File. This site includes information regarding sediment grain size, device power generation, and acoustic effects.

**Accessing Demonstration Files**

To access the demonstration files relevant to this analysis, please refer to the :ref:`tutorial-files-access` section. This demonstration utilizes the :file:`pacwave` and :file:`style_files` folders as detailed in :ref:`DEMO_files`. A comprehensive list of files provided is available below in :ref:`pacwave_files`.

**Preparing Demonstration Files**

At this point you should have prepared the demonstration files as detailed in :ref:`prepare_tutorial_files`. Please follow the instructions in the :ref:`prepare_tutorial_files` section before proceeding with the Tanana River demonstration.

**QuickMapServices**

Results in this tutorial will utilize the QuickMapServices plugin in QGIS. To install this plugin, see :ref:`quick_map_services` to setup in your QGIS instance.


**List of Tutorial Files**

.. _pacwave_files:

.. code-block:: none
   :caption: DEMO PacWave Tutorial Files

   DEMO
   ├───pacwave   
   │   acoustics_module_SEL_199db_threshold.default
   │   acoustics_module_SEL_HFC_173dB_threshold.default
   │   acoustics_module_SEL_LFC_199dB_threshold.default
   │   acoustics_module_SPL_150db_threshold.default
   │   acoustics_module_219dB_threshold.default
   │   all_modules.default
   │   power_modules.default
   │   shear_stress_module_without_grainsize.default
   │   velocity_module.default
   │   velocity_module_without_motility.default
   │
   ├───mec_not_present
   │       trim_sets_flow_inset_allruns.nc
   │
   ├───mec_present
   │       trim_sets_flow_inset_allruns.nc
   │
   ├───paracousti_files
   │       pacwave_3DSPLs_Hw0.5.nc
   │       pacwave_3DSPLs_Hw1.0.nc
   │       pacwave_3DSPLs_Hw1.5.nc
   │       pacwave_3DSPLs_Hw2.0.nc
   │       pacwave_3DSPLs_Hw2.5.nc
   │       pacwave_3DSPLs_Hw3.0.nc
   │       pacwave_3DSPLs_Hw3.5.nc
   │       pacwave_3DSPLs_Hw4.0.nc
   │       pacwave_3DSPLs_Hw4.5.nc
   │       pacwave_3DSPLs_Hw5.0.nc
   │       pacwave_3DSPLs_Hw5.5.nc
   │       pacwave_3DSPLs_Hw6.0.nc
   │       pacwave_3DSPLs_Hw6.5.nc
   │       pacwave_3DSPLs_Hw7.0.nc
   │
   ├───power_files
   │   ├───16x6
   │   │       POWER_ABS_001.OUT
   │   │       POWER_ABS_002.OUT
   │   │       POWER_ABS_003.OUT
   │   │       POWER_ABS_004.OUT
   │   │       POWER_ABS_005.OUT
   │   │       POWER_ABS_006.OUT
   │   │       POWER_ABS_007.OUT
   │   │       POWER_ABS_008.OUT
   │   │       POWER_ABS_009.OUT
   │   │       POWER_ABS_010.OUT
   │   │       POWER_ABS_011.OUT
   │   │       POWER_ABS_012.OUT
   │   │       POWER_ABS_013.OUT
   │   │       POWER_ABS_014.OUT
   │   │       POWER_ABS_015.OUT
   │   │       POWER_ABS_016.OUT
   │   │       POWER_ABS_017.OUT
   │   │       POWER_ABS_018.OUT
   │   │       POWER_ABS_019.OUT
   │   │       POWER_ABS_020.OUT
   │   │       POWER_ABS_021.OUT
   │   │       POWER_ABS_022.OUT
   │   │       POWER_ABS_023.OUT
   │   │       POWER_ABS_024.OUT
   │   │       rect_16x6.obt
   │   │       rect_16x6.pol
   │   │
   │   └───4x4
   │           POWER_ABS_001.OUT
   │           POWER_ABS_002.OUT
   │           POWER_ABS_003.OUT
   │           POWER_ABS_004.OUT
   │           POWER_ABS_005.OUT
   │           POWER_ABS_006.OUT
   │           POWER_ABS_007.OUT
   │           POWER_ABS_008.OUT
   │           POWER_ABS_009.OUT
   │           POWER_ABS_010.OUT
   │           POWER_ABS_011.OUT
   │           POWER_ABS_012.OUT
   │           POWER_ABS_013.OUT
   │           POWER_ABS_014.OUT
   │           POWER_ABS_015.OUT
   │           POWER_ABS_016.OUT
   │           POWER_ABS_017.OUT
   │           POWER_ABS_018.OUT
   │           POWER_ABS_019.OUT
   │           POWER_ABS_020.OUT
   │           POWER_ABS_021.OUT
   │           POWER_ABS_022.OUT
   │           POWER_ABS_023.OUT
   │           POWER_ABS_024.OUT
   │           rect_4x4.obt
   │           rect_4x4.pol
   │
   ├───probabilities
   │       hydrodynamic_probabilities.csv
   │       paracousti_probabilities.csv
   │
   ├───receptor
   │       acoustic_receptor_threshold_100.csv
   │       acoustic_receptor_threshold_120.csv
   │       acoustic_receptor_threshold_219.csv
   │       grainsize_receptor.tif
   │       grain_size_receptor.csv
   │       velocity_receptor.csv
   │
   ├───area_of_interest
   │       habitat_classification.tif
   │
   └───species
         whale_watch_predictions_2021_01.csv
         whale_watch_predictions_2021_02.csv
         whale_watch_predictions_2021_03.csv
         whale_watch_predictions_2021_04.csv
         whale_watch_predictions_2021_05.csv
         whale_watch_predictions_2021_06.csv
         whale_watch_predictions_2021_07.csv
         whale_watch_predictions_2021_08.csv
         whale_watch_predictions_2021_09.csv
         whale_watch_predictions_2021_10.csv
         whale_watch_predictions_2021_11.csv
         whale_watch_predictions_2021_12.csv