.. _index:

.. figure:: media/SEAT_white.png
   :width: 95%
   :align: left
   :alt: SEAT Logo

.. toctree::
   :maxdepth: 4
   :hidden:

   overview.rst
   installation.rst
   tutorials/tutorials.rst
   modules/modules.rst

.. raw:: html

   <div style="clear: both;"></div>


Spatial Environmental Assessment Toolkit (SEAT)
===================================================

The Spatial Environmental Assessment Toolkit (SEAT) is a specialized plugin for QGIS, designed to produce spatial assessment maps and generate statistical CSV files. It offers users the capability to craft risk quantification maps by merging numerical model outcomes with unique site-specific data. This empowers providers with a robust tool to refine array layouts and minimize adverse environmental impacts.

Objective
---------

SEAT's primary goal is to offer a simplified method for pinpointing areas susceptible to alterations due to marine energy developments. To realize this objective, SEAT harnesses open-source models to simulate device parameters within the environment. The analysis of the results is carried out by incorporating site-specific data into a GIS platform. This platform not only renders spatial risk maps but also quantifies, classifies, and facilitates a direct comparison of potential array configurations.

How It Works
------------

SEAT operates as an extension to the open-source spatial analysis tool, QGIS. It processes risk metrics derived from model results, site-specific receptor maps, and user-set thresholds.


Functionality
-------------

SEAT's primary functionality is to enable users to create risk quantification maps. This is achieved by integrating numerical model results with site-specific information. Below, we delve into specific applications of SEAT using Wave Energy Converters (WEC) and Current Energy Converters (CEC) as examples.

1. Current Energy Converters (CECs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Risk Computation**: The risk of environmental change is determined by comparing conditions with marine energy (ME) devices to conditions without them.

**Example Application**: Consider the Tanana River Test Site at Nenana, AK. Here, SEAT visualizes the spatial distribution of shear stress derived from the Dflow-FM model. Using these distributions, the risk to the environment from CECs is computed. This risk assessment leverages the CEC power matrix and the spatial metrics of shear stress and velocity. The cumulative risk for each CEC is then aggregated to present a comprehensive spatial risk distribution.

.. figure:: media/CEC_SEAT.png
   :width: 100%
   :align: center
   :alt: SEAT Application to the Tanana River Test Site at Nenana, AK
   
   SEAT Application to the Tanana River Test Site at Nenana, AK


2. Wave Energy Converters (WECs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Risk Computation:** Changes in environmental conditions, termed as the "stressor layer," are matched with site-specific receptors to compute risk. This change is then scaled by the probability of the forcing condition and aggregated across all considered conditions.

**Example Application**: Consider the PacWave testsite off the coast of Newport, OR. Here, SEAT visualizes the spatial distribution of wave energy flux derived from the SWAN model. Using these distributions, the risk to the environment from WECs is computed. This risk assessment leverages the WEC power matrix and the spatial metrics of wave energy flux, wave height, and wave period. The cumulative risk for each WEC is then aggregated to present a comprehensive spatial risk distribution.

.. figure:: media/WEC_power_matrix.png
   :width: 80%
   :align: center
   :alt: WEC devices represented with Power Matrix relative to HS, TP. 
   
   WEC devices represented with Power Matrix relative to HS, TP.    


.. figure:: media/WEC_spatial.png
  :width: 80%
  :align: center
  :alt: Spatial data such as sediment grainsize, species density, etc.
  
  Spatial data such as sediment grainsize, species density, etc.   


.. figure:: media/WEC_risk.png
   :width: 80%
   :align: center
   :alt: Results represent influence of arrays energy capture across range of conditions and receptor response
   
   Results represent influence of arrays energy capture across range of conditions and receptor response



Contribution 
============

We welcome contributions to SEAT. Please visit our GitHub repository and open an issue or pull request.
https://github.com/sandialabs/seat-qgis-plugin

Contact & Support
==================

Please contact us by opening an issue on GitHub.
https://github.com/sandialabs/seat-qgis-plugin


License
=======

SEAT is distributed under the Revised BSD License. See the LICENSE file in the SEAT repository for the complete license text.
https://github.com/sandialabs/seat-qgis-plugin/blob/main/LICENSE


Acknowledgments
===============
SEAT is developed by Sandia National Laboratories and Integral Consulting. The following individuals have contributed to the development of SEAT:

- **Sandia National Laboratories**

  - Sterling Olson
  - Chris Chartrand
  - Jesse Roberts

- **Integral**
  
  - Tim Nelson
  - Sam McWilliams
  - Craig Jones

Funding
=======

The development of SEAT is funded by the U.S. Department of Energy's Water Power Technologies Office. The copyright of SEAT is held by Sandia National Laboratories. The software is distributed under the Revised BSD License.

Sandia National Laboratories is a multi-mission laboratory managed and operated by National Technology and Engineering Solutions of Sandia, LLC., a wholly-owned subsidiary of Honeywell International, Inc., for the U.S. Department of Energy's National Nuclear Security Administration under contract DE-NA0003525.