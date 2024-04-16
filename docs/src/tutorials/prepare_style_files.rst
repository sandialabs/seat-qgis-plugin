.. _prepare_style_files:

Setting Up Style Files
======================

Below shows the initial contents of the style_files folder. In order to use the demo, 
the \*.default files will need to be converted to \*.csv, and the paths within the \*.default files will need to be updated as explained below.

..  code-block:: none
    :caption: style_files Folder Contents

    DEMO
    └───style_files
        │   PacWave_style_files_all_modules.default
        |   Tanana_style_files_all_modules.default
        └───Layer Style
                acoustics_density_demo.qml
                acoustics_percent_demo.qml
                acoustics_stressor_demo.qml
                acoustics_threshold_demo.qml
                habitat_classification.qml
                receptor_grain_size_blues.qml
                receptor_grain_size_single_250.qml
                shear_stress_continuous.qml
                shear_stress_continuous_unstructured.qml
                shear_stress_receptor_classified.qml
                shear_stress_receptor_continuous.qml
                shear_stress_risk_metric.qml
                velocity_continuous_stressor_vik.qml
                velocity_motility.qml
                velocity_motility_classification_vik.qml



In the tutorial files, we have provided style files to use with SEAT in a folder named ``style_files``. 
Before you can use the style files you need to modify the ``.default`` files to point to the full file path. A python module is available which will update the files for you, or you can do by hand.

Option 1: Python
^^^^^^^^^^^^^^^^^^^^
If you do have python on your machine, there is a python script provided that will adjust the paths and save the files as a csv for you. You can call `adjust_style_path.py` from the command line in the provided Demo files folder: 

.. code-block:: bash
    :caption: adjust_style_path.py

    python adjust_style_path.py
    Please enter the path to the directory containing default files:
    C:\Your\StyleFile\Path\Here\style_files

    Paths in default files in directory C:\Your\StyleFile\Path\Here\style_files have been updated and saved as CSV.

Option 2: Manual
^^^^^^^^^^^^^^^^^^^^

1. Open the style files path and open a ``.default`` file in a text editor.
2. Replace ``<style_folder>`` with ``C:\Your\StyleFile\Path\Here\style_files``. 
3. Replace ``.default`` with ``.csv``. 

   .. figure:: ../media/default_style_files.webp
      :scale: 60 %
      :alt: Edit the highlighted fields with the path and proper file extension.