## Sandia Spatial Environmental Assessment Tool QGIS Plugin
![SEAT logo](icon.png "SEAT Logo")

This repository contains code for the Spatial Environmental Assessment Tool (SEAT) QGIS Plugin. This is a collaboration between Integral Consulting and Sandia National Laboratories.

## Install

To install download the latest [stresser_receptor_calc.zip](https://github.com/IntegralEnvision/SEAT-QGIS-Plugin/releases/latest/download/stressor_receptor_calc.zip) file from  the [Releases](https://github.com/IntegralEnvision/SEAT-QGIS-Plugin/releases/latest) page. You can then use the Install from zip option under the Install Plugins dialog in [QGIS](https://docs.qgis.org/3.22/en/docs/training_manual/qgis_plugins/fetching_plugins.html). 

Plugins are located in C:\Users\<USER>AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\ 

The Python package of [netCDF](https://github.com/Unidata/netcdf4-python) is required. This can be installed by opening "C:\Program Files\QGIS 3.22.6\OSGeo4W.bat" by right - clicking Run As Administrator and running pip install netCDF4.

## Development

Clone the repository and push changes To trigger a release buid use the following 
```
git <tag>
git push origin <tag>
```

In QGIS there are two plugins that are helpful. [Plugin Reloader](https://plugins.qgis.org/plugins/plugin_reloader/) and [FirstAid](https://plugins.qgis.org/plugins/firstaid/)

## Dependencies
* QGIS 3.22.6
* netCDF4
* Python libraries with QGIS 3.22.6
