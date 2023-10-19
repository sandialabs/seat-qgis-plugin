

Setting Up Style Files
======================

In order for the style files to work you will need to set the full path in each of the csv files.

..  code-block:: none
    :caption: style_files Folder Contents

    DEMO
    └───style_files
        │   Acoustics_blue_whale - Copy.csv
        │   Acoustics_blue_whale.csv
        │   Acoustics_fake_whale.csv
        │   Shear_Stress - Structured.csv
        │   Shear_Stress - Unstructured.csv
        │   Velocity.csv
        │
        └───Layer Style
                acoustics_density_demo.qml
                acoustics_percent_demo.qml
                acoustics_stressor_bluewhale.qml
                acoustics_stressor_demo.qml
                acoustics_threshold_demo.qml
                receptor_blues.qml
                shear_stress_continuous.qml
                shear_stress_continuous_unstructured.qml
                shear_stress_receptor_classified.qml
                shear_stress_receptor_continuous.qml
                shear_stress_reclass.qml
                velocity_continuous_stressor_vik.qml
                velocity_continuous_stressor_with_receptor.qml
                velocity_motility_classification_vik.qml


For example if you open **Acoustics_blue_whale.csv** you will see the following:

+-----------------+-----------------------------------------------------------------------------------------------+
| Type            | Style                                                                                         |
+=================+===============================================================================================+
| Stressor        | H:\\Projects\\C1308_SEAT\\SEAT_inputs\\style_files\\Layer Style\\acoustics_stressor_demo.qml  |
+-----------------+-----------------------------------------------------------------------------------------------+
| Threshold       | H:\\Projects\\C1308_SEAT\\SEAT_inputs\\style_files\\Layer Style\\acoustics_threshold_demo.qml |
+-----------------+-----------------------------------------------------------------------------------------------+
| Species Percent | H:\\Projects\\C1308_SEAT\\SEAT_inputs\\style_files\\Layer Style\\acoustics_percent_demo.qml   |
+-----------------+-----------------------------------------------------------------------------------------------+
| Species Density | H:\\Projects\\C1308_SEAT\\SEAT_inputs\\style_files\\Layer Style\\acoustics_density_demo.qml   |
+-----------------+-----------------------------------------------------------------------------------------------+

You will need to set the full path to your current working directory for these to work. E.g. replacing :file:`H:\\Projects\\C1308_SEAT\\SEAT_inputs\\` with the path to your style_files directory.
