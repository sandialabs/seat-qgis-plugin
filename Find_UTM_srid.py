"""

Purpose: finds utm zone for given WGS longitude and latitude
Project: Sandia
Notes: unclear how well this works at high latitudes (+/- 90)
from here: https://lists.osgeo.org/pipermail/postgis-users/2005-December/010253.html

Example usage: find_utm_srid(-124.14, 44, 4326)
2021-03-10. Initially Written. Eben Pendleton

"""

import math


def find_utm_srid(lon, lat, srid):

    """
    Given a WGS 64 srid calculate the corresponding UTM srid

    :param lon: WGS 84 (srid 4326) longitude value)
    :param lat: WGS 84 (srid 4326) latitude value)
    :param srid: WGS 64 srid 4326 to make sure the function is appropriate
    :return: out_srid: UTM srid
    """

    assert srid == 4326, f"find_utm_srid: input geometry has wrong SRID {srid}"

    if lat < 0:
        # south hemisphere
        base_srid = 32700
    else:
        # north hemisphere or on equator
        base_srid = 32600

    # calculate final srid
    out_srid = base_srid + math.floor((lon + 186) / 6)

    if lon == 180:
        out_srid = base_srid + 60

    return out_srid
