.. _prepare_style_files:

Setting Up Style Files
======================

In the tutorial files, we have provided style files to use with SEAT in a folder named ``style_files``. Before you can use the style files you need to modify the ``.default`` files with the full file path to the style files and then save them as csv files. The file extension is set as ``.default`` to prevent the user from loading the file into SEAT without making the necessary adjustments. For example, if you have placed the style files folder in ``C:\Users\USER\Desktop\DEMO\style_files`` you would open each ``.default`` file and replace ``<style_folder>`` with ``C:\Users\USER\Desktop\DEMO\style_files``. 

If you do not have python, you can open the csv files in a text editor and replace ``<style_folder>`` with the path to the style_folder on your machine. If you do have python on your machine, there is a python script provided that will adjust the paths and save the files as a csv for you. You can call `localize_style_path.py` from the command line in the provided Demo files folder: 

.. code-block:: bash
    :caption: localize_style_path.py

    python localize_style_files.py
    Please enter the path to the directory containing default files:
    C:\Users\sterl\OneDrive\Desktop\DEMO\style_files

    Paths in default files in directory C:\Users\sterl\OneDrive\Desktop\DEMO\style_files have been updated and saved as CSV.


The below shows the initial contents of the style_files folder. You will need to convert each of these to  csv files as described above to use them in the demo.

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
