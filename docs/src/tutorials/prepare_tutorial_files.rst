.. _prepare_tutorial_files:

Prepare Tutorial Files
==========================

In the tutorial files, we have provided ``.default`` files which need the full path on your local machine. The file extension is set as ``.default`` to prevent the user from loading the file into SEAT without making the necessary adjustments. 


Using Python
------------ 
.. code-block:: bash
    :caption: localize_tutorial_files.py

    python localize_style_files.py

The above will look for the folders mentioned in the :ref:`tutorial-files-access` (pacwave, tanana, and style_files) and update the paths in the ``.default`` files to match the path on your machine.

Manually
--------

If you do not want to use python you can open each ``.default`` file in a text editor and replace ``<style_folder>`` or ``<input_folder>``  with full path to the location on your machine. After updating you must save the file as an \*.ini file for the Pacwave or Tanana River examples. The style files must be saved as \*.csv. 

You must setup the style file for the Pacwave or Tanana River examples to use the style files. If you only want to run specific Tanana River or Pacwave examples you only need to update the default file of interest. All of the included default files are listed below.

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

For example, if you have placed the style files folder in ``C:\Users\USER\Desktop\DEMO\style_files`` you would open each ``.default`` file and replace ``<style_folder>`` with ``C:\Users\USER\Desktop\DEMO\style_files``. 

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