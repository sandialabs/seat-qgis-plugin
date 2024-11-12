.. _03_species_properties:
Species Properties
--------------------

The species properties section of the SEAT QGIS plugin allows users to specify the species spatial probability/density directory and the species file averaged area.


**Weighting**: The weighting dropdown shows the signal weighted metrics within the device model results files (see Paracousti Pre-Processing). 

.. figure:: ../../media/weighting_dropdown.webp
   :scale: 100 %
   :alt: Dropdown menu showing the signal weighted metrics

**Acoustic Metric**: Unweighted and weighted variables present in the device model results files. Options change depending on weighting selected (see Paracousti Pre-Processing).

Unweighted metrics:

.. figure:: ../../media/acoustic_metrics_dropdown_unweighted.webp
   :scale: 100 %
   :alt: Dropdown menu showing the acoustic metrics for the unweighted version.

Weighted metrics:

.. figure:: ../../media/acoustic_metrics_dropdown_weighted.webp
   :scale: 100 %
   :alt: Dropdown menu showing the acoustic metrics for the weighted version.

**Acoustic Threshold Value**: Threshold above which negative impacts are expected. The units update to match Acoustic Metric selected.

.. figure:: ../../media/acoustic_threshold_value.webp
   :scale: 100 %
   :alt: Input box for the acoustic threshold value.


Species Spatial Probability / Density Directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This input specifies the directory where the
species present and species density files are located. Both files must be located within the same directory.

.. figure:: ../../media/acoustics_species_spatial_probability_density_dir.webp
   :scale: 100 %
   :alt: Secondary Constraint


- **Directory Structure**:

  - The designated directory should contain both the Species Percent and Species Density files.


.. code-block:: none
   :caption: Species Spatial Probability / Density Directory Structure
      
      DEMO
      ├───pacwave
      │   ├───species
      │   │   ├───WhaleWatchPredictions_2021_01.csv
      │   │   ├───WhaleWatchPredictions_2021_02.csv
      │   │   ├───WhaleWatchPredictions_2021_03.csv
      │   │   ├───WhaleWatchPredictions_2021_04.csv


An example of a file is shown below:

.. code-block:: text
   :caption: WhaleWatchPredictions_2021_01.csv

   "","longitude","latitude","bathy","bathyrms","sst","chl","ssh","sshrms","month","year","fitmean","sdfit","percent","density","sddens","upper","lower"
   "1",225,30,-4878.5,145.013092041,19.3042380721481,0.131973730461833,0.10315625,NA,1,2021,NA,NA,NA,NA,NA,NA,NA
   "2",225,30.25,-4845.25,94.5832061768,19.1984631521385,0.139408998412115,0.1158875,NA,1,2021,NA,NA,NA,NA,NA,NA,NA
   "3",225,30.5,-4792,136.986038208,19.1373958299844,0.138623459694399,0.1290125,NA,1,2021,NA,NA,NA,NA,NA,NA,NA
   ...
   "6235",245,48.5,NA,NA,NA,NA,NA,NA,1,2021,NA,NA,NA,NA,NA,NA,NA
   "6236",245,48.75,NA,NA,NA,NA,NA,NA,1,2021,NA,NA,NA,NA,NA,NA,NA
   "6237",245,49,NA,NA,NA,NA,NA,NA,1,2021,NA,NA,NA,NA,NA,NA,NA


**Species File Averaged Area (km^2)**

The grid size of the species percent and density files. This is used for scaling to each ParAcousti grid cell. Leave blank or set to 0 to prevent scaling.


.. figure:: ../../media/species_file_averaged_area.webp
   :scale: 100 %
   :alt: Spatial Probability/Density Grid Resolution