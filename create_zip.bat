@echo off

set TIMESLOT=%date%
echo [%TIMESLOT%] START ...................... create_zip

echo [%TIMESLOT%] COPY ....................... .\seat -> .\seat_qgis_plugin
xcopy ".\seat" ".\seat_qgis_plugin" /E /I

set RELEASE=%TIMESLOT%
echo [%TIMESLOT%] ZIP ........................ %cd%\seat_qgis_plugin_%RELEASE%.zip

:: Requires PowerShell for zip functionality
powershell.exe -Command "Compress-Archive -Path '.\seat_qgis_plugin\*' -DestinationPath '.\seat_qgis_plugin_%RELEASE%.zip'"

echo [%TIMESLOT%] RM ......................... .\seat_qgis_plugin
if exist ".\seat_qgis_plugin" (
    rmdir /s /q ".\seat_qgis_plugin"
) else (
    echo Directory not found: .\seat_qgis_plugin
)

set ENDTIME=%YEAR%_%MONTH%_%DAY%.%HOUR%_%MIN%
echo [%TIMESLOT%] END ........................ create_release
