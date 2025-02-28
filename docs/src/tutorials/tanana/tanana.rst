Tanana River Site
=====================

.. toctree::
   :maxdepth: 1
   :hidden:

   sedimentation.rst
   velocity.rst 

The Tanana River, situated in central Alaska, is being evaluated as a potential site for harnessing current energy. 
The nearby town of Nenana, nestled on the inner bank of a river bend, could significantly benefit from local energy generation as opposed to relying on 
diesel generators. A current energy capture device has undergone testing on-site, and ongoing data collection pertaining to local fish populations and benthic 
characteristics is expected to provide a robust understanding of the long-term impact of device deployment on the ecosystem. 
A model was created on an unstructured grid using DFlow-FM to simulate various flow return periods with device arrays. 
The flow scenarios are examined both under baseline conditions (no devices present) and with devices deployed.

**Accessing Demonstration Files**

To access the demonstration files relevant to this analysis, please refer to the :ref:`tutorial-files-access` section. This demonstration utilizes the :file:`tanana_river` and :file:`style_files` folders as detailed in :ref:`DEMO_files`. A comprehensive list of files provided is available below in :ref:`tanana_river_files`.

**Preparing Demonstration Files**

The demonstration files should be prepared as detailed in :ref:`prepare_tutorial_files`. Please follow the instructions in the :ref:`prepare_tutorial_files` section before proceeding with the Tanana River demonstration.

**QuickMapServices**

Results in this tutorial will utilize the QuickMapServices plugin in QGIS. To install this plugin, see :ref:`quick_map_services` to setup in your QGIS instance.


**List of Files**

.. _tanana_river_files:

..  code-block:: none
  :caption: Tanana River Demo Files

   DEMO
   └───tanana_river
      │   shear_and_velocity_modules_with_receptors.default
      │   shear_and_velocity_modules_without_receptors.default
      │   shear_stress_module_with_receptors.default
      │   shear_stress_module_without_receptors.default
      │   velocity_with_receptors.default
      │   velocity_without_receptors.default
      │   
      ├───mec_not_present
      │       0_tanana_100_map.nc
      │       0_tanana_10_map.nc
      │       0_tanana_1_map.nc
      │       0_tanana_25_map.nc
      │       0_tanana_50_map.nc
      │       0_tanana_5_map.nc
      │
      ├───mec_present
      │       9_tanana_100_map.nc
      │       9_tanana_10_map.nc
      │       9_tanana_1_map.nc
      │       9_tanana_25_map.nc
      │       9_tanana_50_map.nc
      │       9_tanana_5_map.nc
      │
      └───receptor
            grain_size_receptor.csv
            velocity_receptor.csv
