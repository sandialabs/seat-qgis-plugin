::qgis_tests.bat
@echo off

:: Set the QGIS installation path
set "QGIS_ROOT=C:\Program Files\QGIS 3.34.11"

:: Set PYTHONPATH to include QGIS Python modules
set "PYTHONPATH=%QGIS_ROOT%\apps\qgis-ltr\python;%PYTHONPATH%"
set "PYTHONHOME=%QGIS_ROOT%\apps\Python312"

:: Set PATH to include QGIS binaries and libraries
set "PATH=%QGIS_ROOT%\bin;%QGIS_ROOT%\apps\qgis-ltr\bin;%PATH%"

:: Install dependencies
@REM "%QGIS_ROOT%\bin\python.exe"  tests\install_dependencies.py "C:\\Program Files\\QGIS 3.34.9\\bin\\python.exe"

:: Run pytest on all test scripts in the directory
@REM "%QGIS_ROOT%\bin\python.exe" -m pytest .
"%QGIS_ROOT%\bin\python.exe" -m pytest tests\test_stressor_receptor_calc.py
@REM "%QGIS_ROOT%\bin\python.exe" -m pylint seat
@REM "%QGIS_ROOT%\bin\python.exe" -m pylint seat/__init__.py
@REM "%QGIS_ROOT%\bin\python.exe" -m pylint seat/plugin_upload.py