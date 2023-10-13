# Sandia Spatial Environmental Assessment Toolkit QGIS Plugin

<a href='https://github.com/IntegralEnvision/SEAT-QGIS-Plugin'><img src='seat/icon.png' align="right" height="60" /></a>

[![release](https://github.com/IntegralEnvision/seat-qgis-plugin/actions/workflows/release.yaml/badge.svg)](https://github.com/IntegralEnvision/seat-qgis-plugin/actions/workflows/release.yaml)

This repository contains code for the Spatial Environmental Assessment Toolkit (SEAT) QGIS Plugin. This is a collaboration between [Integral Consulting](https://integral-corp.com) and [Sandia National Laboratories](https://www.sandia.gov/).

## Installation

### Requirements

- QGIS >= 3.16
- Python for QGIS >= 3.16
- [netCDF](https://github.com/Unidata/netcdf4-python) >= 3.5.4 - Python install location varies depending on your OS.

  - **Windows** - run `C:\Program Files\QGIS 3.16\OSGeo4W.bat` as administrator.
  - **Linux & MacOS** - Open the Python console in QGIS and enter the commands below to determine where your Python install is:

    ```python
    import sys
    print(sys.exec_prefix)
    ```

### Plugin Install

Download the latest [release](https://github.com/IntegralEnvision/SEAT-QGIS-Plugin/releases/latest) zip file. You can then use the _Install from zip_ option under the [Install Plugins dialog in QGIS](https://docs.qgis.org/3.22/en/docs/training_manual/qgis_plugins/fetching_plugins.html).

The installed plugin is located here:

- **Windows**: `AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins`
- **Linux**: `.local/share/QGIS/QGIS3/profiles/default/python/plugins`
- **Mac** `Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins`

## Development

The codebase was initially generated using the [Qgis-Plugin-Builder](https://g-sherman.github.io/Qgis-Plugin-Builder/). Follow the [PyQGIS Developer Cookbook](https://docs.qgis.org/testing/en/docs/pyqgis_developer_cookbook/index.html) for documentation on developing plugins for QGIS with Python.

There are two QGIS plugins that are helpful during the development process:

- [Plugin Reloader](https://plugins.qgis.org/plugins/plugin_reloader/)
- [FirstAid](https://plugins.qgis.org/plugins/firstaid/)

### Releases

To trigger a release buid on GitHub use the following commands:

```bash
git tag -a <version> -m "<release notes>"
git push --tag
```

## Contents

Below is a brief description of the repository contents.

- [seat-qgis-plugin](https://github.com/IntegralEnvision/seat-qgis-plugin): Repository root.

  - [.github/workflows/release.yaml](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/.github/workflows/release.yaml): Automated GitHub routine that is triggered upon committing a new tag.
  - [.pre-commit-config.yaml](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/.pre-commit-config.yaml): Collection of hooks to run prior to committing the code state to Git. Hooks include linters and code formatters.
  - [create_zip.sh](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/create_zip.sh): Shell script to create a zipped code archive locally, which can be imported to the QGIS plugin installer.
  - [seat-qgis-plugin/seat](https://github.com/IntegralEnvision/seat-qgis-plugin/tree/main/seat): Plugin code.
    - [stressor_receptor_calc.py](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/seat/stressor_receptor_calc.py): Main plugin script for input and display.
    - [stressor_receptor_calc_dialog.py](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/seat/stressor_receptor_calc_dialog.py): Initializes the plugin GUI.
    - [stressor_receptor_calc_dialog_base.ui](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/seat/stressor_receptor_calc_dialog_base.ui): Interface configuration (generated with Qt Designer).
    - [resources.qrc](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/seat/resources.qrc): Application resources such as icons and translation files.
    - [resources.py](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/seat/resources.py): Resource file generated from compiling resources.qrc.
    - [plugin_upload.py](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/seat/plugin_upload.py): Utility - upload a plugin package to the [QGIS plugin repository](https://plugins.qgis.org/plugins/).
    - [pb_tool.cfg](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/seat/pb_tool.cfg): Configuration file for the plugin builder tool.
    - [metadata.txt](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/seat/metadata.txt): Plugin metadata.
    - [icon.png](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/seat/icon.png): Plugin icon.
    - [compile.bat](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/seat/compile.bat): Compile plugin resources (Windows).
    - [shear_stress_module.py](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/seat/shear_stress_module.py): Calculates and generates the shear stress stressor maps and statistics files.
    - [velocity_module.py](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/seat/velocity_module.py): Calculates and generates the velocity stressor maps and statistics files.
    - [acoustics_module.py](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/seat/acoustics_module.py): Calculates and generates the paracousti stressor maps and statistics files.
    - [power_module.py](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/seat/power_module.py): Calculates and generates the wec/cec power generated plots and statistics files.
    - [stressor_utils.py](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/seat/stressor_utils.py): General processing scripts.
