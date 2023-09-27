.. _installtion:

Installation
=====================

Prerequisites
-------------
There is both software and model data that is required to run SEAT. The software requirements are listed below. The model data requirements are listed in the models section.

Model Files
^^^^^^^^^^^
Depending on you application you will need to bring model files and site data to SEAT. Currently, SEAT has built in support for the following models:

- `SNL-Delft3D <https://github.com/sandialabs/SNL-Delft3D-CEC>`_ 

  * Structured & Unstructured Grids

- `SNL-SWAN <https://github.com/sandialabs/SNL-SWAN>`_
- `Paracousti <https://github.com/sandialabs/Paracousti>`_

Software
^^^^^^^^

- `QGIS <https://www.qgis.org/en/site/forusers/download.html>`_ >= 3.16


Download SEAT
--------------

Option 1: Clone
^^^^^^^^^^^^^^^

.. code-block:: sh

   git clone https://github.com/sandialabs/seat-qgis-plugin.git

Option 2: Download
^^^^^^^^^^^^^^^^^^

https://github.com/sandialabs/seat-qgis-plugin
- Download the create_zip.sh




NetCDF4
-------

- run `C:\Program Files\QGIS 3.22.10\OSGeo4W.bat` as administrator.

  * Note replace QGIS 3.22.10 with the installed QGIS version
  * On windows run `C:\Program Files\QGIS 3.16\OSGeo4W.bat` as administrator.
  * Linux & MacOS - Open the Python console in QGIS and enter the commands below to determine where your Python install is:

- Obtain the python path: 

.. code-block:: python

   import sys
   print(sys.exec_prefix)

- Navigate to the python path and “pip install netcdf4”

QGIS
---------------------
- In QGIS, click on the plugins toolbar and select “Manage and Install Plugins”
- Select the Install from ZIP option.

.. figure:: media/installPlugin.png
   :scale: 50 %
   :alt: Install from ZIP

- Navigate to the SEAT zip package.
- Click Install Plugin.
- The SEAT icon should appear in the toolbar.

.. figure:: media/SEAT_Toolbar.png
   :scale: 125 %
   :alt: SEAT icon in QGIS toolbar
   

- And as a Plugin menu option.
