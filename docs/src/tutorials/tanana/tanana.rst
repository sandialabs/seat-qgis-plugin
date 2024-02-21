Tanana River Site
=====================

.. toctree::
   :maxdepth: 1
   :hidden:

   input.rst
   sedimentation.rst
   velocity.rst 

The Tanana River, situated in central Alaska, is being evaluated as a potential site for harnessing current energy. The nearby town of Nenana, nestled on the inner bank of a river bend, could significantly benefit from local energy generation as opposed to relying on diesel generators. A current energy capture device has undergone testing on-site, and ongoing data collection pertaining to local fish populations and benthic characteristics is expected to provide a robust understanding of the long-term impact of device deployment on the ecosystem. A model was created on an unstructured grid using DFlow-FM to simulate various flow return periods with device arrays. The flow scenarios are examined both under baseline conditions (no devices present) and with devices deployed.

**Accessing Demonstration Files**

To access the demonstration files relevant to this analysis, please refer to the :ref:`tutorial-files-access` section. This demonstration utilizes the :file:`DEMO Tanan River` and :file:`style_files` folders as detailed in :ref:`DEMO_files`. A comprehensive list of files contained in the unstructured folder is available in :ref:`tanana_river_files`.


.. _preparing_input_files:

.. rubric:: Preparing Demo Input Files


If you have not done so before, you will need to create input files for the default files provided. To do so, naviage to :file:`DEMO/Tanana River/`. In that folder you will find a number of .default files. You need to open these and replace `<input_folder>` with the path to the DEMO unstructured folder on your machine and ``<style_folder>`` with the path to the style_files folder on your machine. If you use python, a script titled ``localize_input_files.py`` will do this for you. You can run this script by navigating to the DEMO folder in a terminal and typing ``python localize_input_files.py``. If you do not have python, you can open the files in a text editor and replace the text manually or with a find and replace feature. If changing via a text editor, save the file as a .ini file.

Example use of the script is shown below. After running the script, .ini files will appear in the DEMO unstructured folder. These are the files you will use to load the inputs into the SEAT GUI.


.. code-block:: bash
   
   $ python localize_input_files.py 

.. code-block:: none

   Where are your input files? C:\DEMO Files\Tanana River
   Where is your style_files folder? C:\DEMO Files\style_files


**List of Files**

.. _tanana_river_files:

..  code-block:: none
  :caption: Tanana River Demo Files

   DEMO
   └───TananaRiver
      │   Shear_and_Velocity_with_receptor.default
      │   Shear_with_receptor.default
      │   Velocity_with_receptor.default
      │
      ├───tanana_nodev
      │       0_tanana_1_map.nc
      │       0_tanana_5_map.nc
      │       0_tanana_10_map.nc
      │       0_tanana_25_map.nc
      │       0_tanana_50_map.nc
      │       0_tanana_100_map.nc
      │
      ├───tanana_dev_9
      │       9_tanana_1_map.nc
      │       9_tanana_5_map.nc
      │       9_tanana_10_map.nc
      │       9_tanana_25_map.nc
      │       9_tanana_50_map.nc
      │       9_tanana_100_map.nc
      |
      └───receptor
            grain_size_receptor.csv
            velocity_receptor.csv
