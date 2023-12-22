@echo off

:: Set the QGIS installation path
set "QGIS_ROOT=C:\Program Files\QGIS 3.34.1"

:: Set PYTHONPATH to include QGIS Python modules
set "PYTHONPATH=%QGIS_ROOT%\apps\qgis\python;%PYTHONPATH%"
set "PYTHONHOME=%QGIS_ROOT%\apps\Python39"

:: Set PATH to include QGIS binaries and libraries
set "PATH=%QGIS_ROOT%\bin;%QGIS_ROOT%\apps\qgis\bin;%PATH%"

:: Upgrade pip
"%QGIS_ROOT%\bin\python.exe" -m pip install --upgrade pip

:: Check if netCDF4 is installed and install if not
"%QGIS_ROOT%\bin\python.exe" -m pip show netCDF4 >nul 2>&1
if %errorlevel% neq 0 (
    echo netCDF4 not found, installing...
    "%QGIS_ROOT%\bin\python.exe" -m pip install netCDF4 ipdb
)
"%QGIS_ROOT%\bin\python.exe" -m pip install ipdb

:: Run your QGIS-based Python script
"%QGIS_ROOT%\bin\python.exe" test_shear_stress.py
