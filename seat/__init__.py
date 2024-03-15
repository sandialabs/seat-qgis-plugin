# -*- coding: utf-8 -*-
"""
/***************************************************************************.
Name                 : seat
Date                 : 2021-04-19
Contact              : https://github.com/sandialabs/seat-qgis-plugin
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""


def classFactory(iface):  # pylint: disable=invalid-name
    """
    Load StressorReceptorCalc class from file StressorReceptorCalc.
    A QGIS plugin  calculates a response layer from stressor and receptor layers

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .stressor_receptor_calc import StressorReceptorCalc

    return StressorReceptorCalc(iface)
