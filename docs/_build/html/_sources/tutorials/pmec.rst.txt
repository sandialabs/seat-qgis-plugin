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


PMEC power generated per hydrodynamic scenario

+-------+-------+---------+--------------+---------+-----------+---------+---------------------+--------------+
| Hs    | Tp    | Dp      | % of dir bin | % of yr | run order | Exclude | Power_Run_Name      | Power [W]    |
| [m]   | [s]   | [deg]   |              |         |           |         |                     |              |
+=======+=======+=========+==============+=========+===========+=========+=====================+==============+
| 1.76  | 6.6   | 221.8   | 15.41        | 0.39    | 6         |         | POWER_ABS_010.OUT   | 95268714.111 |
+-------+-------+---------+--------------+---------+-----------+---------+---------------------+--------------+
| 2.67  | 8.2   | 223.0   | 40.68        | 1.029   | 16        |         | POWER_ABS_004.OUT   | 7461469.758  |
+-------+-------+---------+--------------+---------+-----------+---------+---------------------+--------------+
| 4.06  | 10.16 | 223.1   | 3.75         | 0.095   | 24 x      | X       | POWER_ABS_021.OUT   | 5567445.958  |
+-------+-------+---------+--------------+---------+-----------+---------+---------------------+--------------+
| 7.05  | 12.33 | 223.6   | 23.04        | 0.586   | 2         |         | POWER_ABS_013.OUT   | 7447026.758  |
+-------+-------+---------+--------------+---------+-----------+---------+---------------------+--------------+
| 2.11  | 11.6  | 223.9   | 8.06         | 0.203   | 20 x      | X       | POWER_ABS_016.OUT   | 10581894.4   |
+-------+-------+---------+--------------+---------+-----------+---------+---------------------+--------------+
| 4.91  | 13.62 | 251.2   | 11.99        | 1.764   | 23        |         | POWER_ABS_019.OUT   | 10851884.9   |
+-------+-------+---------+--------------+---------+-----------+---------+---------------------+--------------+
| 1.7   | 7.73  | 244.8   | 15.03        | 2.212   | 5         |         | POWER_ABS_017.OUT   | 6136127.46   |
+-------+-------+---------+--------------+---------+-----------+---------+---------------------+--------------+
| 2.69  | 9.8   | 248.5   | 26.75        | 3.937   | 17        |         | POWER_ABS_018.OUT   | 16078901.8   |
+-------+-------+---------+--------------+---------+-----------+---------+---------------------+--------------+
| 1.23  | 14.62 | 248.8   | 18.54        | 2.729   | 1         |         | POWER_ABS_023.OUT   | 2841905.7    |
+-------+-------+---------+--------------+---------+-----------+---------+---------------------+--------------+
| 2.31  | 17.54 | 249.4   | 2.72         | 0.584   | 14        |         | POWER_ABS_003.OUT   | 4375975.52   |
+-------+-------+---------+--------------+---------+-----------+---------+---------------------+--------------+
| 2.94  | 11.77 | 250.6   | 23.72        | 3.49    | 7         |         | POWER_ABS_015.OUT   | 7369018.41   |
+-------+-------+---------+--------------+---------+-----------+---------+---------------------+--------------+
| 4.9   | 14.43 | 275.8   | 8.78         | 4.698   | 22 x      | X       | POWER_ABS_020.OUT   | 274670833    |
+-------+-------+---------+--------------+---------+-----------+---------+---------------------+--------------+
| 1.54  | 8.62  | 278     | 11.188       |         | 4         |         | POWER_ABS_011.OUT   | 206397678.2  |
+-------+-------+---------+--------------+---------+-----------+---------+---------------------+--------------+
| 3.66  | 12    | 277.2   | 20.95        | 11.121  | 19 x      | X       | POWER_ABS_022.OUT   | 1149464816   |
+-------+-------+---------+--------------+---------+-----------+---------+---------------------+--------------+
| 2.16  | 10.71 | 277.5   | 25.39        | 13.589  | 12        |         | POWER_ABS_014.OUT   | 138944163.6  |
+-------+-------+---------+--------------+---------+-----------+---------+---------------------+--------------+
| 1.85  | 13.54 | 277.2   | 16.21        | 8.674   | 8         |         | POWER_ABS_009.OUT   | 234556568.9  |
+-------+-------+---------+--------------+---------+-----------+---------+---------------------+--------------+
| 2.05  | 16.51 | 276.4   | 7.77         | 4.159   | 9 x       | X       | POWER_ABS_012.OUT   | 76652472.5   |
+-------+-------+---------+--------------+---------+-----------+---------+---------------------+--------------+
| 2.81  | 11.9  | 298.8   | 18.07        | 8.297   | 7         |         | POWER_ABS_005.OUT   | 275272215.2  |
+-------+-------+---------+--------------+---------+-----------+---------+---------------------+--------------+
| 2.16  | 13.35 | 301.2   | 10.7         | 3.12    | 3         |         | POWER_ABS_006.OUT   | 2934573.29   |
+-------+-------+---------+--------------+---------+-----------+---------+---------------------+--------------+
| 1.81  | 10.22 | 302.6   | 22.06        | 6.664   | 18        |         | POWER_ABS_008.OUT   | 191651685.6  |
+-------+-------+---------+--------------+---------+-----------+---------+---------------------+--------------+
| 2.66  | 11.02 | 297.2   | 26.48        | 7.72    | 15        |         | POWER_ABS_013.OUT   | 161280626.8  |
+-------+-------+---------+--------------+---------+-----------+---------+---------------------+--------------+
| 2.08  | 16.53 | 295.6   | 5.28         | 1.54    | 10        |         | POWER_ABS_007.OUT   | 13757051.64  |
+-------+-------+---------+--------------+---------+-----------+---------+---------------------+--------------+
| 4.65  | 13.23 | 296.2   | 6.22         | 1.813   | 21        |         | POWER_ABS_005.OUT   | 15581540.52  |
+-------+-------+---------+--------------+---------+-----------+---------+---------------------+--------------+



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

   DEMO
   ├───DEMO paracousti
   │   │   demo_paracousti_without_species.default
   │   │   demo_paracousti_without_species.ini
   │   │   demo_paracousti_with_receptor.default
   │   │   demo_paracousti_with_receptor.ini
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
   │   │       Acoustic_Receptor - BlueWhale - Copy.csv
   │   │       Acoustic_Receptor - BlueWhale.csv
   │   │       Acoustic_Receptor - Harassment _Whale-IA-AVD-10.csv
   │   │       Acoustic_Receptor - Harassment _Whale.csv
   │   │
   │   └───species
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
   │
   └───DEMO structured
      │   shear_stress_without_receptor.default
      │   shear_stress_without_receptor.ini    
      │   shear_stress_with_receptor.default   
      │   shear_stress_with_receptor.ini       
      │   velocity_without_receptor.default    
      │   velocity_without_receptor.ini        
      │   velocity_with_receptor.default       
      │   velocity_with_receptor.ini
      │
      ├───devices-not-present
      │       trim_sets_flow_inset_allruns.nc  
      │       trim_sets_flow_inset_allruns.nc.aux.xml
      │
      ├───devices-present
      │       trim_sets_flow_inset_allruns.nc
      │
      ├───Output
      │   └───ShearStress_with_receptor
      │           _20231023.log
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
      │       probabilities.csv
      │
      ├───receptor
      │       grainsize_receptor.tif
      │       grainsize_receptor.tif.aux.xml
      │       grain_size_receptor.csv
      │       velocity_receptor.csv
      │
      └───_plugin-config-files
            cgrant_4x4.ini
            cgrant_test.ini
            oregon_wec_config.ini
            oregon_wec_config_16x6.ini
            oregon_wec_config_4x4.ini
            