# Sandia Spatial Environmental Assessment Tool QGIS Plugin

<a href='https://github.com/IntegralEnvision/SEAT-QGIS-Plugin'><img src='code/icon.png' align="right" height="60" /></a>

[![Release](https://github.com/IntegralEnvision/seat-qgis-plugin/actions/workflows/zip_release.yaml/badge.svg?event=release)](https://github.com/IntegralEnvision/seat-qgis-plugin/actions/workflows/zip_release.yaml)

This repository contains code for the Spatial Environmental Assessment Tool (SEAT) QGIS Plugin. This is a collaboration between [Integral Consulting](https://integral-corp.com) and [Sandia National Laboratories](https://www.sandia.gov/).

## Installation

### Requirements

- QGIS 3.22.6
- Python for QGIS 3.22.6
- [netCDF](https://github.com/Unidata/netcdf4-python) - Python install procedures vary depending on your OS.

  - **Windows** - run `C:\Program Files\QGIS 3.22.6\OSGeo4W.bat` as administrator, then `pip install netCDF4`
  - **Linux & MacOS** - both OS use the system's Python environment, therefore you will need Python (3+) installed on your machine prior to installing QGIS. Using your system Python version, run `pip install netCDF4`. If there are multiple Python versions on your machine, you may need to use the QGIS Python console to determine which install to use:

    ```python
    import sys
    print(sys.exec_prefix)
    ```

### Plugin Install

Download the latest [release](https://github.com/IntegralEnvision/SEAT-QGIS-Plugin/releases/latest) zip file. You can then use the _Install from zip_ option under the [Install Plugins dialog in QGIS](https://docs.qgis.org/3.22/en/docs/training_manual/qgis_plugins/fetching_plugins.html).

The installed plugin is located here:

- **Windows**: `C:\Users\<USER>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\`
- **Linux**: `/home/<USER>/.local/share/QGIS/QGIS3/profiles/default/python/plugins`
- **Mac** `/Users/<USER>/Library/Application Support/QGIS/QGIS3/profiles/default/python`

## Development

The codebase was initially generated using the [Qgis-Plugin-Builder](https://g-sherman.github.io/Qgis-Plugin-Builder/). Follow the [PyQGIS Developer Cookbook](https://docs.qgis.org/testing/en/docs/pyqgis_developer_cookbook/index.html) for documentation on developing plugins for QGIS with Python.

There are two QGIS plugins that are helpful during the development process:

- [Plugin Reloader](https://plugins.qgis.org/plugins/plugin_reloader/)
- [FirstAid](https://plugins.qgis.org/plugins/firstaid/)

### Releases

To trigger a release buid on GitHub use the following commands:

```bash
git <tag>
git push origin <tag>
```

## Notes

Folders in this repository contain all data files and code needed to successfully run the SEAT QGIS Plugin. Below is a brief summary on the contents of each folder in the root directory. There is a separate README within each folder going into more detail.

### [code](./code/)

Code used to create the QGIS plugin.

The steps below outline how to install the plugin in QGIS:

1. Navigate to [https://github.com/IntegralEnvision/SEAT-QGIS-Plugin/releases/latest](https://github.com/IntegralEnvision/SEAT-QGIS-Plugin/releases/latest) and download the file zip file
1. Open QGIS
1. Navigate to `Plugins > Manage and Install Plugins...`. A window will open
1. Click on `Install from ZIP` in the sidebar then enter the path to your zip file downloaded from GitHub
1. Follow the dialog to complete installation

### [plugin-input](./plugin-input)

There are two subdirectories in the input folder, [oregon](./plugin-input/oregon) and [tanana](./plugin-input/tanana). The two directories are test cases for the tool. Both test case directories contain the same set of subdirectories, but the datafiles within them are different.

The subdirecotries directly relate to the plugin input parameters, where each subdirectory is a single input parameter. It is not clear what the parameter data are or should be a this time, but it is what the project staff were provided. At the time of writing this (**_2022-08-05_**), it is not clear if any of the inputs are checked and validated by the tool.

![GUI](./resources/GUI.png)

Plugin input parameter configurations can be saved from the QGIS GUI. Case specific input configurations are saved under `_plugin-config-files` and can be reused by the plugin to simplify parameter selections.

### [plugin-output](./plugin-output)

Files created by the plugin for each case should be saved here and dated. There are subdirectories for each test case output.
