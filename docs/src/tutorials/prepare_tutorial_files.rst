.. _prepare_tutorial_files:

Prepare Tutorial Files
==========================

There are two sets of files that need to be modified before using the demo:

1. Style files
2. Input files

In order to use the demo, the \*.default files will need to be converted to \*.csv, and the paths within the \*.default files will need to be updated. There are two ways to modify the files, explained below.


..  code-block:: none
    :caption: Tutorial file default files

    DEMO 
    ├───pacwave
    │   acoustics_module _100db_threshold.default
    │   acoustics_module_120dB_threshold.default
    │   acoustics_module_219dB_threshold.default
    │   all_modules.default
    │   shear_stress_module.default
    │   shear_stress_module_no_receptor.default
    │   velocity_module.default
    │   velocity_module_no_receptor.default
    │
    ├───style_files
    │   pacwave_style_files_all_modules.default
    │   tanana_style_files_all_modules.default
    │
    └───tanana_river
        shear_and_velocity_with_receptor.default
        shear_with_receptor.default
        velocity_with_receptor.default



Option 1: Python
^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash
    :caption: localize_tutorial_files.py

    python localize_tutorial_files.py
    Where are your input files for pacwave?:
    C:\Your\Path\Here\pacwave


The above will look for the folders mentioned in the :ref:`tutorial-files-access` (pacwave, tanana, and style_files) and update the paths in the ``.default`` files to match the path on your machine.

Option 2: Manually
^^^^^^^^^^^^^^^^^^^^

If you do not want to use python you can open each ``.default`` file in a text editor and replace ``<style_folder>`` or ``<input_folder>``  with full path to the location on your machine. After updating you must save the file as an \*.ini file for the Pacwave or Tanana River examples. The style files must be saved as \*.csv. 



For example, if you have placed the style files folder in ``C:\Your\Path\Here\style_files`` you would open each ``.default`` file and replace ``<style_folder>`` with ``C:\Users\USER\Desktop\DEMO\style_files``. 

..  code-block:: none
    :caption: Style Folder \*.default: Replace <style_folder> save as csv

    Type,Style
    shear_stress_without_devices,<style_folder>\layer_style\shear_stress_continuous.qml
    shear_stress_with_devices,<style_folder>\layer_style\shear_stress_continuous.qml
    shear_stress_difference,<style_folder>\layer_style\shear_stress_continuous.qml
    ...


The below shows the initial contents of the style_files folder. You will need to convert each of these to csv files as described above to use them in the demo.

..  code-block:: none
    :caption: Pacwave or Tanana River \*.default: Replace <input_folder> & <style_folder> save as \*.ini

    [Input]
    shear stress device present filepath = <input_folder>/mec_present
    shear stress device not present filepath = <input_folder>/mec_not_present
    shear stress averaging = Maximum
    ...
    coordinate reference system = 32606
    output style files = <style_folder>/tanana_style_files_all_modules.csv

    [Output]
    output filepath = <input_folder>/Output/Shear_with_receptor