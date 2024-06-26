Species Spatial Probability / Density Directory
------------------------------------------------

In ParAcousti, this input specifies the directory containing: Species Percent and Species Density. Both files must be located within the same directory.

- **Directory Structure**:

  - The designated directory should contain both the Species Percent and Species Density files.

- **File Naming**:

  - Detailed information regarding the naming conventions of these files and the conditions they pertain to can be found in the Model Probabilities File.

.. figure:: ../../media/acoustics_species_spatial_probability_density_dir.webp
   :scale: 100 %
   :alt: Secondary Constraint

**Example:**

Below is example probabilites file from the ParAcousti DEMO Files. The first column is the ParAcousti results file name, the second column is the Species Percent file name, the third column is the Species Density file name, and the fourth column is the percent of the year that the ParAcousti file is applicable. In this example, the Species Percent and Species Density files are the same for all ParAcousti files. 

.. code-block:: text
   :caption: boundary_conditions.csv
   
   Paracousti File,Species Percent Occurance File,Species Density File,% of yr
   PacWave_3DSPLs_Hw0.5.nc,WhaleWatchPredictions_2021_01.csv,WhaleWatchPredictions_2021_01.csv,0
   PacWave_3DSPLs_Hw1.0.nc,WhaleWatchPredictions_2021_02.csv,WhaleWatchPredictions_2021_02.csv,2.729
   PacWave_3DSPLs_Hw1.5.nc,WhaleWatchPredictions_2021_03.csv,WhaleWatchPredictions_2021_03.csv,20.268
   PacWave_3DSPLs_Hw2.0.nc,WhaleWatchPredictions_2021_04.csv,WhaleWatchPredictions_2021_04.csv,39.769
   PacWave_3DSPLs_Hw2.5.nc,WhaleWatchPredictions_2021_05.csv,WhaleWatchPredictions_2021_05.csv,13.27
   PacWave_3DSPLs_Hw3.0.nc,WhaleWatchPredictions_2021_06.csv,WhaleWatchPredictions_2021_06.csv,3.49
   PacWave_3DSPLs_Hw3.5.nc,WhaleWatchPredictions_2021_07.csv,WhaleWatchPredictions_2021_07.csv,11.212
   PacWave_3DSPLs_Hw4.0.nc,WhaleWatchPredictions_2021_08.csv,WhaleWatchPredictions_2021_08.csv,0.593
   PacWave_3DSPLs_Hw4.5.nc,WhaleWatchPredictions_2021_09.csv,WhaleWatchPredictions_2021_09.csv,1.813
   PacWave_3DSPLs_Hw5.0.nc,WhaleWatchPredictions_2021_10.csv,WhaleWatchPredictions_2021_10.csv,6.462
   PacWave_3DSPLs_Hw5.5.nc,WhaleWatchPredictions_2021_11.csv,WhaleWatchPredictions_2021_11.csv,0
   PacWave_3DSPLs_Hw6.0.nc,WhaleWatchPredictions_2021_12.csv,WhaleWatchPredictions_2021_12.csv,0
   PacWave_3DSPLs_Hw6.5.nc,WhaleWatchPredictions_2021_01.csv,WhaleWatchPredictions_2021_01.csv,0
   PacWave_3DSPLs_Hw7.0.nc,WhaleWatchPredictions_2021_02.csv,WhaleWatchPredictions_2021_02.csv,0.086


The Species Percent and Species Density files are located in the directory as specified by the "Secondary Constraint". For example, in the Paracousti Demo files, there is a directory called "species" that contains the Species Percent and Species Density files. The Species Percent and Species Density files have the names as specifed above e.g., "WhaleWatchPredictions_2021_01.csv". Opening one of these files in a text editor shows the following:

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
