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
"""

# pylint: disable=invalid-name
def classFactory(iface):  # Function must be named classFactory for QGIS
    """Load StressorReceptorCalc class from file StressorReceptorCalc.
    A QGIS plugin calculates a response layer from stressor and receptor layers.

    Args:
        iface: A QGIS interface instance.

    Returns:
        StressorReceptorCalc: The initialized plugin class
    """
    # pylint: disable=import-outside-toplevel
    from .stressor_receptor_calc import StressorReceptorCalc
    return StressorReceptorCalc(iface)
