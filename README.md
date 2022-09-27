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
