PMEC Site
----------

The Pacific Marine Energy Center (PMEC) is a designated area for marine energy testing on the coast of Oregon. This site has been the focus of model development and SEAT application. A coupled hydrodynamic and wave model was developed using SNL-SWAN and Delft3D-Flow. A range of site conditions is listed in the Model Probabilities File. This site includes information regarding sediment grain size, device power generation, and acoustic effects.

Accessing Demonstration Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To access the demonstration files relevant to this analysis, please refer to the section :ref:`tutorial-files-access`. This demonstration utilizes the :file:`DEMO unstructured` and :file:`style_files` folders as detailed in :ref:`DEMO_files`. A comprehensive list of files contained in the unstructured folder is available in :ref:`structured_files`.

Sedimentation Analysis and Power Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Example model input can be found in "DEMO structured/shear_stress_with_receptor.ini"

This set of inputs includes a GeoTiff of grain sizes as a receptor layer, power generation files at .OUT files with georeferencing in .pol files. The model files are concatenated into a single .nc file.

* The sedimentation analysis indicates a predominant decrease in sediment erosion and increase in sediment deposition in the lee of the array, with less mobility occurring over larger sediment size classes.

.. figure:: ../media/wec_shear_tutorial.webp
   :scale: 100 %
   :alt: PMEC shear stress example risk

* The power generation is saved as individual images and tables in the selected output folder

.. figure:: ../media/Total_Scaled_Power_per_Device_.webp
   :scale: 100 %
   :alt: PMEC power generated per device bar plots

.. figure:: ../media/Device_Power.webp
   :scale: 100 %
   :alt: PMEC power generated per device heat map

.. figure:: ../media/power_output_csv.webp
   :scale: 100 %
   :alt: PMEC power generated per hydrodynamic scenario


Acoustic Effects
^^^^^^^^^^^^^^^^

The acoustic effects from the WEC array at PMEC can be evaluated using the Acoustic module in the SEAT GUI. This module reads in individual Paracousti model .nc files that correspond to wave conditions. 

For a given probability of occurrence of each wave condition the combined annual acoustic effects can be estimated. SEAT generates a similar stressor layer consisting of the difference between the acoustic effects with and without the array. With a provided receptor file which consists of information regarding the species, threshold value, weighting, and variable used, a threshold map is generated as a percentage of time (based on the probability distribution) that a threshold will be exceeded. For demonstration purposes, an artificially low threshold is used to generate the percent exceeded threshold figure below.

.. figure:: ../media/paracousti_stressor.webp
   :scale: 100 %
   :alt: PMEC sound pressure level stressor

.. figure:: ../media/paracousti_percent.webp
   :scale: 100 %
   :alt: PMEC acoustic threshold exceedance percentage


List of Files
^^^^^^^^^^^^^

.. _structured_files:

..  code-block:: none
  :caption: DEMO unstructured Directory 
  :emphasize-lines: 2, 45

   DEMO
   ├───DEMO paracousti
   │   │   demo_paracousti_without_species.ini
   │   │   demo_paracousti_with_receptor.ini
   │   │   demo_paracousti_with_receptor_FakeWhale - Copy.ini
   │   │
   │   ├───paracousti_files
   │   │       PacWave_3DSPLs_Hw0.5.nc
   │   │       PacWave_3DSPLs_Hw1.0.nc
   │   │       PacWave_3DSPLs_Hw1.5.nc
   │   │       PacWave_3DSPLs_Hw2.0.nc
   │   │       PacWave_3DSPLs_Hw2.5.nc
   │   │       PacWave_3DSPLs_Hw3.0.nc
   │   │       PacWave_3DSPLs_Hw3.5.nc
   │   │       PacWave_3DSPLs_Hw4.0.nc
   │   │       PacWave_3DSPLs_Hw4.5.nc
   │   │       PacWave_3DSPLs_Hw5.0.nc
   │   │       PacWave_3DSPLs_Hw5.5.nc
   │   │       PacWave_3DSPLs_Hw6.0.nc
   │   │       PacWave_3DSPLs_Hw6.5.nc
   │   │       PacWave_3DSPLs_Hw7.0.nc
   │   │
   │   ├───probability
   │   │       boundary_conditions.csv
   │   │
   │   ├───receptor
   │   │       Acoustic_Receptor - BlueWhale.csv
   │   │       Acoustic_Receptor - FakeWhale.csv
   │   │
   │   └───species
   │           WhaleWatchPredictions_2009_1.csv
   │           WhaleWatchPredictions_2009_9.csv
   │           WhaleWatchPredictions_2015_04.csv
   │           WhaleWatchPredictions_2015_05.csv
   │           WhaleWatchPredictions_2015_06.csv
   │           WhaleWatchPredictions_2015_08.csv
   │           WhaleWatchPredictions_2015_09.csv
   │           WhaleWatchPredictions_2015_10.csv
   │           WhaleWatchPredictions_2015_11.csv
   │           WhaleWatchPredictions_2015_12.csv
   │           WhaleWatchPredictions_2016_01.csv
   │           WhaleWatchPredictions_2016_02.csv
   │           WhaleWatchPredictions_2016_03.csv
   │           WhaleWatchPredictions_2016_04.csv
   │           WhaleWatchPredictions_2016_05.csv
   │           WhaleWatchPredictions_2016_06.csv
   │           WhaleWatchPredictions_2016_07.csv
   │           WhaleWatchPredictions_2016_08.csv
   │           WhaleWatchPredictions_2016_09.csv
   │           WhaleWatchPredictions_2016_10.csv
   │           WhaleWatchPredictions_2016_11.csv
   │           WhaleWatchPredictions_2016_12.csv
   │           WhaleWatchPredictions_2017_01.csv
   │           WhaleWatchPredictions_2017_02.csv
   │           WhaleWatchPredictions_2017_03.csv
   │           WhaleWatchPredictions_2017_04.csv
   │           WhaleWatchPredictions_2017_05.csv
   │           WhaleWatchPredictions_2017_06.csv
   │           WhaleWatchPredictions_2017_07.csv
   │           WhaleWatchPredictions_2017_08.csv
   │           WhaleWatchPredictions_2017_09.csv
   │           WhaleWatchPredictions_2017_10.csv
   │           WhaleWatchPredictions_2017_11.csv
   │           WhaleWatchPredictions_2017_12.csv
   │           WhaleWatchPredictions_2018_01.csv
   │           WhaleWatchPredictions_2018_02.csv
   │           WhaleWatchPredictions_2018_03.csv
   │           WhaleWatchPredictions_2018_04.csv
   │           WhaleWatchPredictions_2018_05.csv
   │           WhaleWatchPredictions_2018_06.csv
   │           WhaleWatchPredictions_2018_07.csv
   │           WhaleWatchPredictions_2018_08.csv
   │           WhaleWatchPredictions_2018_09.csv
   │           WhaleWatchPredictions_2018_10.csv
   │           WhaleWatchPredictions_2018_11.csv
   │           WhaleWatchPredictions_2018_12.csv
   │           WhaleWatchPredictions_2019_01.csv
   │           WhaleWatchPredictions_2019_02.csv
   │           WhaleWatchPredictions_2019_03.csv
   │           WhaleWatchPredictions_2019_04.csv
   │           WhaleWatchPredictions_2019_05.csv
   │           WhaleWatchPredictions_2019_06.csv
   │           WhaleWatchPredictions_2019_07.csv
   │           WhaleWatchPredictions_2019_08.csv
   │           WhaleWatchPredictions_2019_09.csv
   │           WhaleWatchPredictions_2019_10.csv
   │           WhaleWatchPredictions_2019_11.csv
   │           WhaleWatchPredictions_2019_12.csv
   │           WhaleWatchPredictions_2020_01.csv
   │           WhaleWatchPredictions_2020_02.csv
   │           WhaleWatchPredictions_2020_03.csv
   │           WhaleWatchPredictions_2020_04.csv
   │           WhaleWatchPredictions_2020_05.csv
   │           WhaleWatchPredictions_2020_06.csv
   │           WhaleWatchPredictions_2020_07.csv
   │           WhaleWatchPredictions_2020_09.csv
   │           WhaleWatchPredictions_2020_10.csv
   │           WhaleWatchPredictions_2020_11.csv
   │           WhaleWatchPredictions_2020_12.csv
   │           WhaleWatchPredictions_2021_01.csv
   │           WhaleWatchPredictions_2021_02.csv
   │           WhaleWatchPredictions_2021_03.csv
   │           WhaleWatchPredictions_2021_04.csv
   │           WhaleWatchPredictions_2021_05.csv
   │           WhaleWatchPredictions_2021_06.csv
   │           WhaleWatchPredictions_2021_07.csv
   │           WhaleWatchPredictions_2021_08.csv
   │           WhaleWatchPredictions_2021_09.csv
   │           WhaleWatchPredictions_2021_10.csv
   │           WhaleWatchPredictions_2021_11.csv
   │           WhaleWatchPredictions_2021_12.csv
   │           WhaleWatchPredictions_2022_01.csv
   │           WhaleWatchPredictions_2022_02.csv
   │           WhaleWatchPredictions_2022_03.csv
   │           WhaleWatchPredictions_2022_04.csv
   │           WhaleWatchPredictions_2022_05.csv
   │           WhaleWatchPredictions_2022_06.csv
   │           WhaleWatchPredictions_2022_07.csv
   │           WhaleWatchPredictions_2022_08.csv
   │           WhaleWatchPredictions_2022_09.csv
   │           WhaleWatchPredictions_2022_10.csv
   │           WhaleWatchPredictions_2022_11.csv
   │           WhaleWatchPredictions_2023_01.csv
   │           WhaleWatchPredictions_2023_02.csv
   │           WhaleWatchPredictions_2023_03.csv
   │           WhaleWatchPredictions_2023_04.csv
   │
   ├───DEMO structured
   │   │   shear_stress_without_receptor.ini
   │   │   shear_stress_with_receptor.ini
   │   │   velocity_without_receptor.ini
   │   │   velocity_with_receptor.ini
   │   │
   │   ├───boundary-condition
   │   │       boundary-conditions.csv
   │   │
   │   ├───devices-not-present
   │   │       trim_sets_flow_inset_allruns.nc
   │   │       trim_sets_flow_inset_allruns.nc.aux.xml
   │   │
   │   ├───devices-present
   │   │       trim_sets_flow_inset_allruns.nc
   |   |
   │   ├───power_files
   │   │   ├───16x6
   │   │   │       POWER_ABS_001.OUT
   │   │   │       POWER_ABS_002.OUT
   │   │   │       POWER_ABS_003.OUT
   │   │   │       POWER_ABS_004.OUT
   │   │   │       POWER_ABS_005.OUT
   │   │   │       POWER_ABS_006.OUT
   │   │   │       POWER_ABS_007.OUT
   │   │   │       POWER_ABS_008.OUT
   │   │   │       POWER_ABS_009.OUT
   │   │   │       POWER_ABS_010.OUT
   │   │   │       POWER_ABS_011.OUT
   │   │   │       POWER_ABS_012.OUT
   │   │   │       POWER_ABS_013.OUT
   │   │   │       POWER_ABS_014.OUT
   │   │   │       POWER_ABS_015.OUT
   │   │   │       POWER_ABS_016.OUT
   │   │   │       POWER_ABS_017.OUT
   │   │   │       POWER_ABS_018.OUT
   │   │   │       POWER_ABS_019.OUT
   │   │   │       POWER_ABS_020.OUT
   │   │   │       POWER_ABS_021.OUT
   │   │   │       POWER_ABS_022.OUT
   │   │   │       POWER_ABS_023.OUT
   │   │   │       POWER_ABS_024.OUT
   │   │   │       rect_16x6.obt
   │   │   │       rect_16x6.pol
   │   │   │
   │   │   └───4x4
   │   │           POWER_ABS_001.OUT
   │   │           POWER_ABS_002.OUT
   │   │           POWER_ABS_003.OUT
   │   │           POWER_ABS_004.OUT
   │   │           POWER_ABS_005.OUT
   │   │           POWER_ABS_006.OUT
   │   │           POWER_ABS_007.OUT
   │   │           POWER_ABS_008.OUT
   │   │           POWER_ABS_009.OUT
   │   │           POWER_ABS_010.OUT
   │   │           POWER_ABS_011.OUT
   │   │           POWER_ABS_012.OUT
   │   │           POWER_ABS_013.OUT
   │   │           POWER_ABS_014.OUT
   │   │           POWER_ABS_015.OUT
   │   │           POWER_ABS_016.OUT
   │   │           POWER_ABS_017.OUT
   │   │           POWER_ABS_018.OUT
   │   │           POWER_ABS_019.OUT
   │   │           POWER_ABS_020.OUT
   │   │           POWER_ABS_021.OUT
   │   │           POWER_ABS_022.OUT
   │   │           POWER_ABS_023.OUT
   │   │           POWER_ABS_024.OUT
   │   │           rect_4x4.obt
   │   │           rect_4x4.pol
   │   │
   │   ├───receptor
   │   │       grainsize_receptor.tif
   │   │       grainsize_receptor.tif.aux.xml
   │   │       grain_size_receptor.csv
   │   │       velocity_receptor.csv
   │   │
   │   ├───run-order
   │   │       run_order_wecs.csv
   │   │
   │   └───_plugin-config-files
   │           cgrant_4x4.ini
   │           cgrant_test.ini
   │           oregon_wec_config.ini
   │           oregon_wec_config_16x6.ini
   │           oregon_wec_config_4x4.ini
   │
   └───style_files
      │   Acoustics_blue_whale - Copy.csv
      │   Acoustics_blue_whale.csv
      │   Acoustics_fake_whale.csv
      │   Shear_Stress - Structured.csv
      │   Shear_Stress - Unstructured.csv
      │   Velocity.csv
      │
      └───Layer Style
               acoustics_density_demo.qml
               acoustics_percent_demo.qml
               acoustics_stressor_bluewhale.qml
               ...
               velocity_continuous_stressor_with_receptor.qml
               velocity_motility_classification_vik.qml
