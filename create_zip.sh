#!/bin/bash
#
#   create_zip.sh
#
#   PURPOSE: Create a local zipped code archive which
#       can be used to install the SEAT plugin in QGIS.
#
#   DATE: 2022-08-12
#
#   AUTHOR: Caleb Grant, Integral Consulting Inc.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TIMESLOT=`date +%Y_%m_%d.%H_%M`

printf "[$TIMESLOT] START ...................... create_zip\n"

printf "[$TIMESLOT] COPY ....................... ./seat -> ./seat_qgis_plugin\n"

cp -r ./seat ./seat_qgis_plugin

RELEASE=`date +%Y%m%d`

printf "[$TIMESLOT] ZIP ........................ ${PWD}/seat_qgis_plugin_$RELEASE.zip\n"

zip -r ./seat_qgis_plugin_$RELEASE.zip ./seat_qgis_plugin #> /dev/null 2>&1

printf "[$TIMESLOT] RM ......................... ./seat_qgis_plugin\n"

rm -rf ./seat_qgis_plugin/

ENDTIME=`date +%Y_%m_%d.%H_%M`

printf "[$TIMESLOT] END ........................ create_release\n"
