# Sandia Spatial Environmental Assessment Tool QGIS Plugin

<a href='https://github.com/IntegralEnvision/SEAT-QGIS-Plugin'><img src='code/icon.png' align="right" height="60" /></a>

[![release](https://github.com/IntegralEnvision/seat-qgis-plugin/actions/workflows/release.yaml/badge.svg)](https://github.com/IntegralEnvision/seat-qgis-plugin/actions/workflows/release.yaml)

This repository contains code for the Spatial Environmental Assessment Tool (SEAT) QGIS Plugin. This is a collaboration between [Integral Consulting](https://integral-corp.com) and [Sandia National Laboratories](https://www.sandia.gov/).

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
  - [seat-qgis-plugin/code](https://github.com/IntegralEnvision/seat-qgis-plugin/tree/main/code): Plugin code.
    - [stressor_receptor_calc.py](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/code/stressor_receptor_calc.py): Main plugin script doing the heavy lifting.
    - [stressor_receptor_calc_dialog.py](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/code/stressor_receptor_calc_dialog.py): Initializes the plugin GUI.
    - [stressor_receptor_calc_dialog_base.ui](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/code/stressor_receptor_calc_dialog_base.ui): Interface configuration (generated with Qt Designer).
    - [resources.qrc](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/code/resources.qrc): Application resources such as icons and trnalsation files.
    - [resources.py](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/code/resources.py): Resource file generated from compiling resources.qrc.
    - [readnetcdf_createraster.py](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/code/readnetcdf_createraster.py): Utility - plot normalized comparison of simulations with WECs and without WECs for user selected vairable for all boundary conditions.
    - [plugin_upload.py](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/code/plugin_upload.py): Utility - upload a plugin package to the [QGIS plugin repository](https://plugins.qgis.org/plugins/).
    - [pb_tool.cfg](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/code/pb_tool.cfg): Configuration file for the plugin builder tool.
    - [metadata.txt](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/code/metadata.txt): Plugin metadata.
    - [icon.png](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/code/icon.png): Plugin icon.
    - [compile.bat](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/code/compile.bat): Compile plugin resources (Windows).
    - [Find_UTM_srid.py](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/code/Find_UTM_srid.py): Utility - finds UTM zone for given WGS latitude and longitude.
    - [seat-qgis-plugin/code/inputs](https://github.com/IntegralEnvision/seat-qgis-plugin/tree/main/code/inputs): Files used to configure layer styling.
      - [TaumaxStyle1.csv](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/code/inputs/TaumaxStyle1.csv): Style file configuration for the Taumax calculation.
      - [VelStyle1.csv](https://github.com/IntegralEnvision/seat-qgis-plugin/blob/main/code/inputs/VelStyle1.csv): Style file configuration for the Vel calculation.
      - [seat-qgis-plugin/code/inputs/Layer Style](https://github.com/IntegralEnvision/seat-qgis-plugin/tree/main/code/inputs/Layer%20Style): Layer style files used to set attributes like layer colors, categories, and resampling methods.