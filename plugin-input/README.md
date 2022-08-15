# plugin-input

There are two subdirectories in this folder, [oregon](./plugin-input/oregon) and [tanana](./plugin-input/tanana). The two directories are test cases for the plugin. Both test case directories contain the same set of subdirectories, but the datafiles within them are different.

## Input Data Directories

### 1. \_plugin-config-files

Contains configurations that map plugin parameters to files on disk. Each configuration should map to files within the directories listed below. If they do not, move them into this test case directory to keep things organized.

### 2. boundary-condition

Excel file of model runs and associated probabilities.

### 3. devices-not-present

Model results with marine energy devices **_not_** present. Accepts a single NetCDF file or collection of files.

### 4. devices-present

Model results with marine energy devices present. Accepts a single NetCDF file or collection of files.

### 5. receptor

Raster layer of receptors.

### 6. run-order

Run order reference file.

### 7. secondary-constraint

Unclear what this is or should be.
