Probabilities (optional)
------------------------

The probabilities file, while optional, serves to enhance your analysis, providing insights into the run order and the likelihood of each model condition unfolding. Adhere to the prescribed naming convention (as delineated in the device/baseline model section). Note that this file correlates with the return interval in years.

.. figure:: ../../media/probabilities_input.webp
   :scale: 100 %
   :alt: Interface depicting the Probabilities Input in SEAT's GUI.

For ParAcousti analyses, the file format differs from that of shear stress and velocity.

**Required Columns**:

  1. ParAcousti File: Name of the ParAcousti .nc file.
  2. Species Percent Occurrence File: Accepts either a .csv or .tif format.
  3. Species Density File: Also, either a .csv or .tif format.
  4. % of yr: Denotes the percentage of the year.

**File Specifications**:

- If you're using .csv for the Species Percent Occurrence and Species Density Files, they must contain the essential columns: "latitude", "longitude", and either "percent" and/or "density". All supplementary columns will be overlooked.
- If you opt for a .tif format for the aforementioned files, ensure consistency in the EPSG code across them.

**Example of a ParAcousti Input**

+--------------------------+-----------------------------+------------------------+---------+
| ParAcousti File          | Species % Occurrence File   | Species Density File   | % of yr |
+--------------------------+-----------------------------+------------------------+---------+
| paracousti_data1.nc      | species_occurrence1.csv     | species_density1.csv   | 0.25    |
+--------------------------+-----------------------------+------------------------+---------+
| paracousti_data2.nc      | species_occurrence2.tif     | species_density2.tif   | 0.50    |
+--------------------------+-----------------------------+------------------------+---------+
| paracousti_data3.nc      | species_occurrence3.csv     | species_density3.csv   | 0.10    |
+--------------------------+-----------------------------+------------------------+---------+
| paracousti_data4.nc      | species_occurrence4.tif     | species_density4.tif   | 0.15    |
+--------------------------+-----------------------------+------------------------+---------+

Key:

- `ParAcousti File`: The name of the ParAcousti .nc file.
- `Species % Occurrence File`: Either a .csv or .tif file indicating species percent occurrence.
- `Species Density File`: Either a .csv or .tif file detailing species density.
- `% of yr`: Represents the percentage of the year.
