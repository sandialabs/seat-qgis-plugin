ParAcousti
^^^^^^^^^^

The ParAcousti file format is distinct from formats used for shear stress and velocity.

Required Columns:

  1. ParAcousti File: The name of the ParAcousti .nc file.
  2. Species Percent Occurrence File: This can be either a .csv or .tif file.
  3. Species Density File: This can also be a .csv or .tif file.
  4. % of yr: This represents the percentage value of the year.

File Details:

The Species Percent Occurrence File and Species Density File, when in .csv format, must have the mandatory columns "latitude", "longitude", and either "percent" and/or "density". All other columns will be disregarded.

If using a .tif format for the Species Percent Occurrence File and Species Density File, ensure that the file has the same EPSG code.