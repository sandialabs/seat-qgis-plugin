Acoustics
===========

The ParAcousti module examines acoustic propagation and its thresholds. It compares ParAcousti baseline model runs to model runs with devices present. 
The user has the option to pre-process the ParAcousti results by weighting functions, which emphasizes certain frequencies and de-emphasizes certain frequencies within the 
acoustic signal, depending on if there is a species of interest.   

The module has two tabs: the `inputs <01_inputs>_` tab and the `species properties <03_species_properties>_` tab.

.. figure:: ../../media/acoustics_inputs.webp
   :scale: 100 %
   :alt: Interface to set Velocity stressor in SEAT's GUI.

.. toctree::
   :maxdepth: 1
   :hidden:

   01_inputs.rst
   02_probabilities.rst
   03_species_properties.rst
   04_paracousti_preprocessing.rst
   05_risk_layer.rst
   06_depth_avg.rst