::qgis_tests.bat
@echo off

:: Set the QGIS installation path
set "QGIS_ROOT=C:\Program Files\QGIS 3.34.1"

:: Set PYTHONPATH to include QGIS Python modules
set "PYTHONPATH=%QGIS_ROOT%\apps\qgis\python;%PYTHONPATH%"
set "PYTHONHOME=%QGIS_ROOT%\apps\Python39"

:: Set PATH to include QGIS binaries and libraries
set "PATH=%QGIS_ROOT%\bin;%QGIS_ROOT%\apps\qgis\bin;%PATH%"

:: Install dependencies
"%QGIS_ROOT%\bin\python.exe"  install_dependencies.py "C:\\Program Files\\QGIS 3.34.1\\bin\\python.exe"

:: Run pytest on all test scripts in the directory
"%QGIS_ROOT%\bin\python.exe" -m pytest .
