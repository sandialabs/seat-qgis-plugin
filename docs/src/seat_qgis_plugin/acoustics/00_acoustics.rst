Acoustics
===========

The :ref:`acoustics_module` examines acoustic propagation and its thresholds. It compares ParAcousti baseline model runs to model runs with devices present. 
If the user is interested in analyzing the acoustic effects on a specific species of interest, there is the option to pre-process the ParAcousti results by
using a routine that applies weighting functions to the acoustic signal, detailed in :ref:`01_paracousti_preprocessing`. 


The module has two tabs: the :ref:`02_inputs` tab and the :ref:`03_species_properties` tab.

.. figure:: ../../media/acoustics_inputs.webp
   :scale: 100 %
   :alt: Interface to set Velocity stressor in SEAT's GUI.

.. toctree::
   :maxdepth: 1
   :hidden:

   01_paracousti_preprocessing.rst
   02_inputs.rst
   03_species_properties.rst