.. _prepare_tutorial_files:

Prepare Tutorial Files
==========================

There are two sets of files that need to be modified before using the demo:

1. Style files
2. Input files

..  code-block:: none
    :caption: Tutorial file default files

    DEMO 
    ├───pacwave (input files)
    │   acoustics_module_SEL_199db_threshold.default
    │   acoustics_module_SEL_HFC_173dB_threshold.default
    │   acoustics_module_SEL_LFC_199dB_threshold.default
    │   acoustics_module_SPL_150dB_threshold.default
    │   all_modules.default
    │   power_module.default
    │   shear_stress_module.default
    │   shear_stress_module_without_grainsize.default
    │   velocity_module.default
    │   velocity_module_without_motility.default
    │
    ├───style_files (style files)
    │   pacwave_style_files_all_modules.default
    │   pacwave_style_files_SEL_flat.default
    │   pacwave_style_files_SEL_HFC.default
    │   pacwave_style_files_SEL_LFC.default
    │   pacwave_style_files_SPL.default
    │   tanana_style_files_all_modules.default
    │
    └───tanana_river (input files)
        shear_and_velocity_modules_with_receptors.default
        shear_and_velocity_modules_without_receptors.default
        shear_stress_module_with_receptors.default
        shear_stress_module_without_receptors.default
        velocity_stress_module_with_receptors.default
        velocity_stress_module_without_receptors.default

The ``\*.default`` files point to paths on the local machine. 

In order to use the demo, paths within the ``\*.default`` files need to be updated to point to the correct location. 
Then, the ``\*.default`` files then need to be converted to ``\*.csv`` if it's a style file or ``\*.ini`` if it's an input file. 



Option 1: Python
^^^^^^^^^^^^^^^^^^^^
The localize_tutorial_files.py will look for the folders in the :ref:`DEMO_files` and update the paths in the  ``.default`` files
to match the paths on your machine.

.. code-block:: bash
    :caption: localize_tutorial_files.py

    python localize_tutorial_files.py


Option 2: Manually
^^^^^^^^^^^^^^^^^^^^

If you do not want to use python you can update the files manually. 

1)  Update style files 
""""""""""""""""""""""""

Within the style files directory, open each ``.default`` file in a text editor and replace ``<style_folder>`` with full path to the location on your machine. 
Then, save the style file as a csv.

For example, if the style files folder is located at ``C:\Users\USER\Desktop\DEMO\style_files`` , open the ``.default`` files and replace ``<style_folder>`` with ``C:\Users\USER\Desktop\DEMO\style_files``. 

..  code-block:: none
    :caption: Style Folder \*.default: Replace <style_folder> save as csv

    Type,Style
    shear_stress_without_devices,<style_folder>\layer_style\shear_stress_continuous.qml
    shear_stress_with_devices,<style_folder>\layer_style\shear_stress_continuous.qml
    shear_stress_difference,<style_folder>\layer_style\shear_stress_continuous.qml
    ...

Then, save the ``.default`` file as a ``.csv``

..  code-block:: none
    :caption: Save output files with ``.csv`` file extension

    ├───style_files (style files)
    │   pacwave_style_files_all_modules.csv
    │   tanana_style_files_all_modules.csv
    

2) Update input files  
"""""""""""""""""""""

Within the input files directory, open each ``.default`` file in a text editor and replace ``<input_folder>`` or ``<style_folder>`` with full path to the correct locations on your machine. 


..  code-block:: none
    :caption: Pacwave or Tanana River \*.default: Replace <input_folder> & <style_folder> 

    [Input]
    shear stress device present filepath = <input_folder>/mec_present
    shear stress device not present filepath = <input_folder>/mec_not_present
    shear stress averaging = Maximum
    ...
    coordinate reference system = 32606
    output style files = <style_folder>/tanana_style_files_all_modules.csv

    [Output]
    output filepath = <input_folder>/Output/Shear_with_receptor

Then, save the input file as a ``.ini``.

..  code-block:: none
    :caption: Save output files with ``.ini`` file extension

    └───tanana_river (input files)
        shear_and_velocity_with_receptor.ini
        shear_with_receptor.ini
        velocity_with_receptor.ini
    
    
