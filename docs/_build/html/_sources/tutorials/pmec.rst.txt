PMEC Site
----------

The Pacific Marine Energy Center (PMEC) is a designated area for marine energy testing on the coast of Oregon. This site has been the focus of model development and SEAT application. A coupled hydrodynamic and wave model was developed using SNL-SWAN and Delft3D-Flow. A range of site conditions is listed in the Model Probabilities File. This site includes information regarding sediment grain size, device power generation, and acoustic effects.

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