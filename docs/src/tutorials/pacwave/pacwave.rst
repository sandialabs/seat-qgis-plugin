
PacWave Site
------------

.. toctree::
   :maxdepth: 1
   :hidden:

   sedimentation.rst
   acoustics.rst 

The PacWave is a designated area for marine energy testing on the coast of Oregon. This site has been the focus of model development and SEAT application. A coupled hydrodynamic and wave model was developed using SNL-SWAN and Delft3D-Flow. A range of site conditions is listed in the Model Probabilities File. This site includes information regarding sediment grain size, device power generation, and acoustic effects.

**Accessing Demonstration Files**

To access the demonstration files relevant to this analysis, please refer to the section :ref:`tutorial-files-access`. This demonstration utilizes the :file:`DEMO paracousti`, :file:`DEMO structured,` and :file:`style_files` folders as detailed in :ref:`DEMO_files`. A comprehensive list of files contained in the paracousti and structured folders is available in :ref:`pmec_files`.



**List of Tutorial Files**

.. _pacwave_files:

.. code-block:: none
   :caption: DEMO PacWave Tutorial Files

   DEMO
   ├───PacWave   
   │   │   acoustics_module - 100db threshold.default
   │   │   acoustics_module - 120dB threshold.default
   │   │   acoustics_module - 219dB threshold.default
   │   │   all_modules.default
   │   │   shear_stress_module - no receptor.default
   │   │   shear_stress_module.default
   │   │   velocity module - no receptor.default
   │   │   velocity module.default
   │   │      
   │   ├───MEC not present
   │   │       trim_sets_flow_inset_allruns.nc
   │   │   
   │   ├───MEC present
   │   │       trim_sets_flow_inset_allruns.nc
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
   │   ├───power_files
   │   │   ├───16x6
   |   │   │       POWER_ABS_001.OUT
   |   │   │       POWER_ABS_002.OUT
   |   │   │       POWER_ABS_003.OUT
   |   │   │       POWER_ABS_004.OUT
   |   │   │       POWER_ABS_005.OUT
   |   │   │       POWER_ABS_006.OUT
   |   │   │       POWER_ABS_007.OUT
   |   │   │       POWER_ABS_008.OUT
   |   │   │       POWER_ABS_009.OUT
   |   │   │       POWER_ABS_010.OUT
   |   │   │       POWER_ABS_011.OUT
   |   │   │       POWER_ABS_012.OUT
   |   │   │       POWER_ABS_013.OUT
   |   │   │       POWER_ABS_014.OUT
   |   │   │       POWER_ABS_015.OUT
   |   │   │       POWER_ABS_016.OUT
   |   │   │       POWER_ABS_017.OUT
   |   │   │       POWER_ABS_018.OUT
   |   │   │       POWER_ABS_019.OUT
   |   │   │       POWER_ABS_020.OUT
   |   │   │       POWER_ABS_021.OUT
   |   │   │       POWER_ABS_022.OUT
   |   │   │       POWER_ABS_023.OUT
   |   │   │       POWER_ABS_024.OUT
   |   │   │       rect_16x6.obt
   |   │   │       rect_16x6.pol
   |   │   │
   |   │   └───4x4
   |   │           POWER_ABS_001.OUT
   |   │           POWER_ABS_002.OUT
   |   │           POWER_ABS_003.OUT
   |   │           POWER_ABS_004.OUT
   |   │           POWER_ABS_005.OUT
   |   │           POWER_ABS_006.OUT
   |   │           POWER_ABS_007.OUT
   |   │           POWER_ABS_008.OUT
   |   │           POWER_ABS_009.OUT
   |   │           POWER_ABS_010.OUT
   |   │           POWER_ABS_011.OUT
   |   │           POWER_ABS_012.OUT
   |   │           POWER_ABS_013.OUT
   |   │           POWER_ABS_014.OUT
   |   │           POWER_ABS_015.OUT
   |   │           POWER_ABS_016.OUT
   |   │           POWER_ABS_017.OUT
   |   │           POWER_ABS_018.OUT
   |   │           POWER_ABS_019.OUT
   |   │           POWER_ABS_020.OUT
   |   │           POWER_ABS_021.OUT
   |   │           POWER_ABS_022.OUT
   |   │           POWER_ABS_023.OUT
   |   │           POWER_ABS_024.OUT
   |   │           rect_4x4.obt
   |   │           rect_4x4.pol
   │   │
   │   ├───probabilities
   │   │       hydrodynamic_probabilities.csv
   │   │       paracousti_probabilities.csv
   │   │
   │   ├───receptor
   │   │       Acoustic_Receptor - threshold_100.csv
   │   │       Acoustic_Receptor - threshold_120.csv
   │   │       Acoustic_Receptor - threshold_219.csv
   |   |       grain_size_receptor.csv
   |   |       grainsize_receptor.tif
   |   |       velocity_receptor.csv
   │   │
   │   └───risk layer
   |   |       habitat_classification.tif
   │   │
   │   └───species
       |       WhaleWatchPredictions_2021_01.csv
       |       WhaleWatchPredictions_2021_02.csv
       |       WhaleWatchPredictions_2021_03.csv
       |       WhaleWatchPredictions_2021_04.csv
       |       WhaleWatchPredictions_2021_05.csv
       |       WhaleWatchPredictions_2021_06.csv
       |       WhaleWatchPredictions_2021_07.csv
       |       WhaleWatchPredictions_2021_08.csv
       |       WhaleWatchPredictions_2021_09.csv
       |       WhaleWatchPredictions_2021_10.csv
       |       WhaleWatchPredictions_2021_11.csv
       |       WhaleWatchPredictions_2021_12.csv