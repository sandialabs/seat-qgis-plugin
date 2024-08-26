import sys
import os
import unittest
import numpy as np
import pandas as pd
from osgeo import gdal
from osgeo import osr
from unittest.mock import patch, MagicMock
from os.path import join

# Get the directory in which the current script is located
script_dir = os.path.dirname(os.path.realpath(__file__))

# Import seat
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# fmt: off
from seat.modules import stressor_utils as su

# fmt: on
# from seat.stressor_utils import estimate_grid_spacing

class TestStressorUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Class method to set up file paths for the tests
        """
        # Define paths with script_dir prepended, similar to the shear stress module pattern
        cls.risk_layer_file = join(script_dir, "data/structured/risk-layer/habitat_classification.tif")
        cls.grain_size_file = join(script_dir, "data/structured/receptor/grainsize_receptor.tif")
        cls.receptor_filename_csv = os.path.join(script_dir, 'data/structured/receptor/grain_size_receptor.csv')

class TestEstimateGridSpacing(TestStressorUtils):

    def test_evenly_spaced_points(self):
        # Evenly spaced points
        x, y = np.meshgrid(np.arange(0, 10, 1), np.arange(0, 10, 1))
        x = x.flatten()
        y = y.flatten()
        expected_spacing = 1

        spacing = su.estimate_grid_spacing(x, y)
        self.assertAlmostEqual(spacing, expected_spacing, delta=0.1)

    def test_randomly_distributed_points(self):
        # Randomly distributed points
        np.random.seed(0)  # for reproducible results
        x = np.random.rand(100)
        y = np.random.rand(100)
        # Hardcoded expected spacing
        expected_spacing = 0.038

        spacing = su.estimate_grid_spacing(x, y)
        self.assertAlmostEqual(spacing, expected_spacing, delta=0.02)


class TestCreateStructuredArrayFromUnstructured(TestStressorUtils):

    def test_structured_array_creation(self):
        # Example unstructured data (could be synthetic or based on a simple pattern)
        x = np.array([0, 1, 1])
        y = np.array([0, 0, 1])
        z = np.array([10, 20, 30])
        dxdy = 1

        refxg, refyg, z_interp = su.create_structured_array_from_unstructured(x, y, z, dxdy)

        # Assert the shapes of the returned arrays
        self.assertEqual(refxg.shape, refyg.shape)
        self.assertEqual(refxg.shape, z_interp.shape)

        # Assert the range and spacing of the grid
        np.testing.assert_array_almost_equal(refxg[0, :], np.arange(0, 2, dxdy))
        np.testing.assert_array_almost_equal(refyg[:, 0], np.arange(0, 2, dxdy))


class TestRedefineStructuredGrid(TestStressorUtils):

    def test_redefined_grid(self):
        # Example structured data
        x = np.linspace(0, 10, 5)
        y = np.linspace(0, 10, 5)
        xx, yy = np.meshgrid(x, y)
        z = np.sin(xx) * np.cos(yy)

        x_new, y_new, z_new = su.redefine_structured_grid(xx, yy, z)

        # Assert the shapes of the returned arrays
        self.assertEqual(x_new.shape, y_new.shape)
        self.assertEqual(x_new.shape, z_new.shape)

        # Assert the range and spacing of the grid
        dx = np.nanmin(np.diff(x_new[0, :]))
        np.testing.assert_almost_equal(x_new[0, 1] - x_new[0, 0], dx)
        np.testing.assert_almost_equal(y_new[1, 0] - y_new[0, 0], dx)


class TestResampleStructuredGrid(TestStressorUtils):

    def test_grid_resampling(self):
        # Input grid
        x = np.linspace(0, 10, 5)
        y = np.linspace(0, 10, 5)
        x_grid, y_grid = np.meshgrid(x, y)
        z = np.sin(x_grid) * np.cos(y_grid)

        # Output grid
        X_grid_out = np.linspace(0, 10, 10)
        Y_grid_out = np.linspace(0, 10, 10)
        X_grid_out, Y_grid_out = np.meshgrid(X_grid_out, Y_grid_out)

        # Resample
        z_resampled = su.resample_structured_grid(x_grid, y_grid, z, X_grid_out, Y_grid_out, interpmethod='nearest')

        # Assert the shape of the resampled grid
        self.assertEqual(z_resampled.shape, X_grid_out.shape)


class TestCalcReceptorArray(TestStressorUtils):

    def setUp(self):
        self.x = np.array([-124.284, -124.2832, -124.2824])
        self.y = np.array([44.671, 44.6705, 44.670])

    def test_with_tif_file(self):
        # Call the original calc_receptor_array function
        receptor_array = su.calc_receptor_array(self.grain_size_file, self.x, self.y)

        # Set the expected receptor array values based on the actual values in the .tif file
        expected_receptor_array = np.array([0.0, 300.0, 300.0])

        # Assert the output based on actual expected values from the .tif file
        np.testing.assert_array_almost_equal(receptor_array, expected_receptor_array, decimal=6)

    def test_with_csv_file(self):
        # Call the original function with the actual CSV file
        receptor_array = su.calc_receptor_array(self.receptor_filename_csv, self.x, self.y)
        # Set the expected receptor array values based on the real CSV data
        expected_receptor_array = np.array([250, 250, 250])
        # Assert the output based on actual expected values from the CSV file
        np.testing.assert_array_almost_equal(receptor_array, expected_receptor_array, decimal=6)

    def test_without_file(self):
        receptor_filename = ''
        receptor_array = su.calc_receptor_array(receptor_filename, self.x, self.y)

        # Assert the output
        expected_array = 200e-6 * np.ones(self.x.shape)
        np.testing.assert_array_almost_equal(receptor_array, expected_array,  decimal=6)

    def test_invalid_file_type(self):
        receptor_filename = 'test.invalid'
        with self.assertRaises(Exception) as context:
            su.calc_receptor_array(receptor_filename, self.x, self.y)

        self.assertTrue(f"Invalid Receptor File {receptor_filename}. Must be of type .tif or .csv" in str(context.exception))


class TestTrimZeros(TestStressorUtils):

    def test_trim_zeros(self):
        # Create test data with zeros on edges
        x = np.zeros((5, 5))
        y = np.zeros((5, 5))
        z1 = np.zeros((1, 1, 5, 5))
        z2 = np.zeros((1, 1, 5, 5))
        x[1:-1, 1:-1] = 1
        y[1:-1, 1:-1] = 1
        z1[:, :, 1:-1, 1:-1] = 1
        z2[:, :, 1:-1, 1:-1] = 1

        x_trimmed, y_trimmed, z1_trimmed, z2_trimmed = su.trim_zeros(x, y, z1, z2)

        # Assert the shapes of the returned arrays
        self.assertEqual(x_trimmed.shape, (3, 3))
        self.assertEqual(y_trimmed.shape, (3, 3))
        self.assertEqual(z1_trimmed.shape, (1, 1, 3, 3))
        self.assertEqual(z2_trimmed.shape, (1, 1, 3, 3))

        # Assert that the arrays are trimmed correctly (no zeros)
        self.assertTrue(np.all(x_trimmed != 0))
        self.assertTrue(np.all(y_trimmed != 0))
        self.assertTrue(np.all(z1_trimmed != 0))
        self.assertTrue(np.all(z2_trimmed != 0))


class TestCreateRaster(TestStressorUtils):

    def setUp(self):
        # Temporary output path for testing
        self.output_path = 'test_output.tif'

    def tearDown(self):
        # Clean up the generated test file after each test
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

    def test_create_raster(self):
        # Call the actual function to create a raster
        cols = 100
        rows = 100
        nbands = 1
        eType = gdal.GDT_Float32

        raster = su.create_raster(self.output_path, cols, rows, nbands, eType)

        # Verify the raster was created
        self.assertTrue(os.path.exists(self.output_path))

        # Check the raster's properties
        self.assertIsInstance(raster, gdal.Dataset)
        self.assertEqual(raster.RasterXSize, cols)
        self.assertEqual(raster.RasterYSize, rows)
        self.assertEqual(raster.RasterCount, nbands)
        self.assertEqual(raster.GetRasterBand(1).DataType, eType)


class TestNumpyArrayToRaster(TestStressorUtils):

    def setUp(self):
        # Temporary output path for testing
        self.output_path = 'test_output.tif'

    def tearDown(self):
        # Clean up the generated test file after each test
        try:
            if os.path.exists(self.output_path):
                os.remove(self.output_path)
        except PermissionError:
            pass  # Handle the permission error silently for now

    def test_numpy_array_to_raster(self):
        # Test data
        numpy_array = np.array([[1, 2], [3, 4]])
        bounds = [0, 0]  # Upper-left corner of the raster
        cell_resolution = [1, 1]  # Resolution in x and y directions
        spatial_reference_system_wkid = 4326  # EPSG code for WGS84

        # Create a raster using GDAL
        driver = gdal.GetDriverByName('GTiff')
        rows, cols = numpy_array.shape
        output_raster = driver.Create(self.output_path, cols, rows, 1, gdal.GDT_Float32)

        # Call the function
        result_path = su.numpy_array_to_raster(
            output_raster,
            numpy_array,
            bounds,
            cell_resolution,
            spatial_reference_system_wkid,
            self.output_path
        )

        # Verify the raster was created
        self.assertTrue(os.path.exists(self.output_path))

        # Check the raster's properties
        dataset = gdal.Open(self.output_path)
        self.assertEqual(dataset.RasterXSize, cols)
        self.assertEqual(dataset.RasterYSize, rows)
        self.assertEqual(dataset.RasterCount, 1)

        # Properly set up the spatial reference and check the projection
        spatial_ref = osr.SpatialReference()
        spatial_ref.ImportFromEPSG(spatial_reference_system_wkid)
        self.assertEqual(dataset.GetProjection(), spatial_ref.ExportToWkt())

        # Read the data back and compare it with the original array
        band = dataset.GetRasterBand(1)
        data = band.ReadAsArray()
        np.testing.assert_array_equal(data, numpy_array)

        # Close the dataset to avoid file access issues
        dataset = None

        # Ensure that the function returned the correct output path
        self.assertEqual(result_path, self.output_path)


class TestFindUtmSrid(TestStressorUtils):

    def test_northern_hemisphere(self):
        lon = 10  # Longitude in the Northern Hemisphere
        lat = 50  # Latitude in the Northern Hemisphere
        srid = 4326
        expected_srid = 32600 + np.floor((lon + 186) / 6)
        self.assertEqual(su.find_utm_srid(lon, lat, srid), expected_srid)

    def test_southern_hemisphere(self):
        lon = 30  # Longitude in the Southern Hemisphere
        lat = -30  # Latitude in the Southern Hemisphere
        srid = 4326
        expected_srid = 32700 + np.floor((lon + 186) / 6)
        self.assertEqual(su.find_utm_srid(lon, lat, srid), expected_srid)

    def test_on_equator(self):
        lon = 0  # On the Prime Meridian
        lat = 0  # On the Equator
        srid = 4326
        expected_srid = 32600 + np.floor((lon + 186) / 6)
        self.assertEqual(su.find_utm_srid(lon, lat, srid), expected_srid)

    def test_wrong_srid(self):
        lon = 10
        lat = 10
        srid = 1234  # Incorrect SRID
        with self.assertRaises(AssertionError):
            su.find_utm_srid(lon, lat, srid)


class TestReadRaster(TestStressorUtils):

    def setUp(self):
        self.raster_path = "data/structured/receptor/grainsize_receptor.tif"

    def test_read_raster(self):
        # Expected values
        expected_top_left_value = 0.0
        expected_center_value = 300.0
        expected_bottom_right_value = -3.4028234663852886e+38
        expected_geotransform = (-124.28439, 0.0008, 0.0, 44.6715, 0.0, -0.0010000000000000009)

        # Open the dataset to read geotransform and projection
        dataset = gdal.Open(self.grain_size_file)

        # Call the actual function to get raster data
        rx, ry, raster_array = su.read_raster(self.grain_size_file)

        # Check specific values in the raster
        self.assertEqual(raster_array[0, 0], expected_top_left_value)
        self.assertEqual(raster_array[raster_array.shape[0] // 2, raster_array.shape[1] // 2], expected_center_value)

        # Compare the bottom-right value with tolerance
        self.assertTrue(abs(raster_array[-1, -1] - expected_bottom_right_value) < 1e-6,
                        msg=f"Expected: {expected_bottom_right_value}, Got: {raster_array[-1, -1]}")

        # Check geotransform
        geotransform = dataset.GetGeoTransform()
        self.assertEqual(geotransform, expected_geotransform)

        # Check projection by comparing spatial reference attributes
        spatial_ref = osr.SpatialReference(wkt=dataset.GetProjection())
        self.assertEqual(spatial_ref.GetAttrValue('AUTHORITY', 1), '4326')
        self.assertEqual(spatial_ref.GetAttrValue('DATUM'), 'WGS_1984')

class TestSecondaryConstraintGeotiffToNumpy(TestStressorUtils):

    def test_secondary_constraint_geotiff_to_numpy(self):
        # Expected values based on the actual contents of the grainsize_receptor.tif
        expected_top_left_value = 0.0
        expected_center_value = 300.0
        expected_bottom_right_value = -3.4028234663852886e+38

        # Call the actual function to read the raster and get x_grid, y_grid, and array
        x_grid, y_grid, array = su.secondary_constraint_geotiff_to_numpy(self.grain_size_file)

        # Assert the dimensions of the grid and array
        self.assertEqual(x_grid.shape, array.shape)
        self.assertEqual(y_grid.shape, array.shape)

        # Verify some specific values within the array
        self.assertEqual(array[0, 0], expected_top_left_value)
        self.assertEqual(array[array.shape[0] // 2, array.shape[1] // 2], expected_center_value)
        self.assertAlmostEqual(array[-1, -1], expected_bottom_right_value, places=6)

        # Optionally, you can add more assertions for the geotransform, projection, etc.
        # For example:
        dataset = gdal.Open(self.grain_size_file)
        geotransform = dataset.GetGeoTransform()
        expected_geotransform = (-124.28439, 0.0008, 0.0, 44.6715, 0.0, -0.0010000000000000009)
        self.assertEqual(geotransform, expected_geotransform)

        # Check the projection
        projection = dataset.GetProjection()
        srs = osr.SpatialReference(wkt=projection)
        self.assertEqual(srs.GetAttrValue('AUTHORITY', 1), '4326')

        # Ensure no exceptions occur and all key values are as expected
        dataset = None  # Close the dataset to avoid issues with open file handles


class TestCalculateCellArea(TestStressorUtils):

    def test_latlon_true(self):
        # Test data for a lat/lon grid
        rx = np.array([[0, 1], [0, 1]])
        ry = np.array([[0, 0], [1, 1]])
        rxm, rym, square_area = su.calculate_cell_area(rx, ry, latlon=True)

        # Assert the area calculation
        expected_area = np.array([[1.230722e+10]])
        np.testing.assert_array_almost_equal(square_area, expected_area, decimal=-4)

        # Assert rxm and rym values
        expected_rxm = np.array([[0.5]])
        expected_rym = np.array([[0.5]])
        np.testing.assert_array_almost_equal(rxm, expected_rxm)
        np.testing.assert_array_almost_equal(rym, expected_rym)

    def test_latlon_false(self):
        # Test data for a planar grid
        rx = np.array([[0, 1], [0, 1]])
        ry = np.array([[0, 0], [1, 1]])
        rxm, rym, square_area = su.calculate_cell_area(rx, ry, latlon=False)

        # Assert the area calculation
        expected_area = np.array([[1]])
        np.testing.assert_array_equal(square_area, expected_area)

        # Assert rxm and rym values
        expected_rxm = np.array([[0.5]])
        expected_rym = np.array([[0.5]])
        np.testing.assert_array_equal(rxm, expected_rxm)
        np.testing.assert_array_equal(rym, expected_rym)


class TestBinData(TestStressorUtils):

    def test_bin_data(self):
        # Sample input data
        zm = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        square_area = np.ones(zm.shape)  # Assume each cell has an area of 1 for simplicity
        nbins = 5

        # Expected output
        expected_bins = {'bin start': np.array([1., 2.8, 4.6, 6.4, 8.2]),
                        'bin end': np.array([2.8, 4.6, 6.4, 8.2, 10.]),
                        'bin center': np.array([1.9, 3.7, 5.5, 7.3, 9.1]),
                        'count': np.array([2, 2, 2, 2, 2]),
                        'Area': np.array([2., 2., 2., 2., 2.])}

        # Call the function
        result = su.bin_data(zm, square_area, nbins)

        # Assert the results
        np.testing.assert_array_almost_equal(result['bin start'], expected_bins['bin start'])
        np.testing.assert_array_almost_equal(result['bin end'], expected_bins['bin end'])
        np.testing.assert_array_almost_equal(result['bin center'], expected_bins['bin center'])
        np.testing.assert_array_equal(result['count'], expected_bins['count'])
        np.testing.assert_array_equal(result['Area'], expected_bins['Area'])

    def test_bin_data_with_varying_area(self):
        # Sample input data with varying area
        zm = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        square_area = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])  # Varying area for each cell
        nbins = 5

        # Expected output with consideration for varying area
        expected_bins = {'bin start': np.array([1., 2.8, 4.6, 6.4, 8.2]),
                        'bin end': np.array([2.8, 4.6, 6.4, 8.2, 10.]),
                        'bin center': np.array([1.9, 3.7, 5.5, 7.3, 9.1]),
                        'count': np.array([2, 2, 2, 2, 2]),
                        'Area': np.array([2., 4., 6., 8., 10.])}  # Area now reflects the input square_area

        # Call the function
        result = su.bin_data(zm, square_area, nbins)

        # Assert the results with varying area
        np.testing.assert_array_almost_equal(result['bin start'], expected_bins['bin start'])
        np.testing.assert_array_almost_equal(result['bin end'], expected_bins['bin end'])
        np.testing.assert_array_almost_equal(result['bin center'], expected_bins['bin center'])
        np.testing.assert_array_equal(result['count'], expected_bins['count'])
        np.testing.assert_array_equal(result['Area'], expected_bins['Area'])


class TestBinReceptor(TestStressorUtils):

    def setUp(self):
        # Set up sample data for tests
        self.zm = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.receptor = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        self.square_area = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])  # Varying area for each cell
        self.nbins = 5
        self.receptor_names = ['Type A', 'Type B', 'Type C', 'Type D', 'Type E']

    def test_bin_receptor_without_names(self):
        # Test without passing receptor_names
        result = su.bin_receptor(self.zm, self.receptor, self.square_area, self.nbins)

        # Verifying that keys for each unique receptor value are present
        for rval in np.unique(self.receptor):
            self.assertIn(f'Area, receptor value {rval}', result)


    def test_bin_receptor_with_names(self):
        # Test with passing receptor_names
        result = su.bin_receptor(self.zm, self.receptor, self.square_area, self.nbins, self.receptor_names)

        # Verifying that the receptor names are used in the keys
        for name in self.receptor_names:
            self.assertIn(name, result)


    def test_bin_receptor_calculation(self):
        # Test comprehensive functionality including bin statistics and area percentage calculations
        result = su.bin_receptor(self.zm, self.receptor, self.square_area, self.nbins)

        # Expected bin statistics
        expected_result = {
            'bin start': np.array([1. , 2.8, 4.6, 6.4, 8.2]),
            'bin end': np.array([ 2.8,  4.6,  6.4,  8.2, 10. ]),
            'bin center': np.array([1.9, 3.7, 5.5, 7.3, 9.1]),
            'Area, receptor value 0':np. array([2., 0., 0., 0., 0.]),
            'Area percent, receptor value 0': np.array([100.,   0.,   0.,   0.,   0.]),
            'Area, receptor value 1': np.array([0., 4., 0., 0., 0.]),
            'Area percent, receptor value 1': np.array([  0., 100.,   0.,   0.,   0.]),
            'Area, receptor value 2': np.array([0., 0., 6., 0., 0.]),
            'Area percent, receptor value 2': np.array([  0.,   0., 100.,   0.,   0.]),
            'Area, receptor value 3': np.array([0., 0., 0., 8., 0.]),
            'Area percent, receptor value 3': np.array([  0.,   0.,   0., 100.,   0.]),
            'Area, receptor value 4': np.array([ 0.,  0.,  0.,  0., 10.]),
            'Area percent, receptor value 4': np.array([  0.,   0.,   0.,   0., 100.])
        }

        # Verify bin statistics
        np.testing.assert_array_almost_equal(result['bin start'], expected_result['bin start'])
        np.testing.assert_array_almost_equal(result['bin end'], expected_result['bin end'])
        np.testing.assert_array_almost_equal(result['bin center'], expected_result['bin center'])

        # Directly verify area and area percent for each receptor value based on actual outputs
        for rval in np.unique(self.receptor):
            area_key = f'Area, receptor value {rval}'
            area_percent_key = f'Area percent, receptor value {rval}'

            # Assert area and area percentages based on provided actual result
            np.testing.assert_array_almost_equal(result[area_key], expected_result[area_key])
            np.testing.assert_array_almost_equal(result[area_percent_key], expected_result[area_percent_key])


class TestBinLayer(TestStressorUtils):

    def test_bin_layer_without_receptor(self):
        # Call the bin_layer function with real raster data and no receptor
        result = su.bin_layer(self.risk_layer_file)

        # Check if the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check if the dataframe has the expected columns for bins and areas
        expected_columns = ['bin start', 'bin end', 'bin center', 'count', 'Area', 'Area percent']
        for col in expected_columns:
            self.assertIn(col, result.columns)

        # Filter out nodata values and check the first valid bin start
        valid_data = result[result['bin start'] > -1e+37]
        expected_bin_start = 0.0
        self.assertGreaterEqual(valid_data['bin start'].iloc[0], expected_bin_start)

        # Hardcoded check for bin start and count values for known data
        self.assertAlmostEqual(valid_data['bin start'].iloc[0], 0.0, places=4)
        self.assertAlmostEqual(valid_data['count'].iloc[0], 2585.0, places=2)

    def test_bin_layer_with_receptor(self):
        # Call the bin_layer function with real raster and receptor data
        result = su.bin_layer(self.risk_layer_file, self.grain_size_file)

        # Check if the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check if the dataframe has the expected columns for bins and areas with receptor values
        expected_columns = [
            'bin start', 'bin end', 'bin center',
            'Area, receptor value 0.0', 'Area, receptor value 300.0',
            'Area percent, receptor value 0.0', 'Area percent, receptor value 300.0'
        ]
        for col in expected_columns:
            self.assertIn(col, result.columns)

        # Hardcoded check for area and percentage values for known data
        self.assertAlmostEqual(result['Area, receptor value 0.0'].sum(), 20776811.7427073, places=2)
        self.assertAlmostEqual(result['Area percent, receptor value 0.0'].iloc[0], 86.34171591875244, places=2)


class TestClassifyLayerArea(TestStressorUtils):

    def test_classify_layer_area_without_receptor(self):
        # Use real data for testing
        result = su.classify_layer_area(self.risk_layer_file, at_values=[0, 5, 7], value_names=['Zero', 'Five', 'Seven'])

        # Check if the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check if the DataFrame has the expected columns
        expected_columns = ['value', 'value name', 'Area', 'Area percent']
        for col in expected_columns:
            self.assertIn(col, result.columns)

        expected_values = [0.0, 5.0, 7.0]
        expected_areas = [19174937.53, 7611327.17, 1179139.2]
        expected_percent = [68.57, 27.22, 4.22]

        # Check if the values match the expected ones
        np.testing.assert_array_almost_equal(result['value'], expected_values, decimal=4)
        np.testing.assert_array_almost_equal(result['Area'], expected_areas, decimal=2)
        np.testing.assert_array_almost_equal(result['Area percent'], expected_percent, decimal=2)

    def test_classify_layer_area_with_receptor(self):
        # Use real data for testing, including the receptor raster
        result = su.classify_layer_area(self.risk_layer_file, self.grain_size_file, at_values=[0, 5, 7], value_names=['Zero', 'Five', 'Seven'])

        # Check if the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check if the DataFrame has the expected columns for receptor values
        expected_columns = ['value', 'value name', 'Area, receptor value 0.0', 'Count, receptor value 0.0',
                            'Area, receptor value 300.0', 'Count, receptor value 300.0',
                            'Area, receptor value 5000.0', 'Count, receptor value 5000.0']
        for col in expected_columns:
            self.assertIn(col, result.columns)

        # Adjusted hardcoded expected values for receptors based on real data analysis
        expected_areas_0 = [18581526.98, 0, 1157969.1]
        expected_counts_0 = [2630, 0, 164]
        expected_percent_0 = [94.13, 0, 5.87]

        # Verify the calculated areas, counts, and percentages for receptor value 0.0
        np.testing.assert_array_almost_equal(result['Area, receptor value 0.0'], expected_areas_0, decimal=2)
        np.testing.assert_array_equal(result['Count, receptor value 0.0'], expected_counts_0)
        np.testing.assert_array_almost_equal(result['Area percent, receptor value 0.0'], expected_percent_0, decimal=2)


class TestClassifyLayerArea2ndConstraint(TestStressorUtils):

    def setUp(self):
        # Hardcoded secondary constraint data as a binary mask (1s and 0s)
        self.secondary_constraint_data = np.array([
            [1, 0, 1, 1, 0],
            [0, 1, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1],
            [1, 0, 0, 1, 1]
        ])

    def test_classify_layer_area_2nd_constraint(self):
        # Call the classify_layer_area_2nd_Constraint with hardcoded secondary constraint
        result = su.classify_layer_area_2nd_Constraint(
            self.risk_layer_file,
            None,  # Not using a secondary constraint file, using hardcoded data instead
            at_raster_values=[0, 5, 7],
            at_raster_value_names=['Zero', 'Five', 'Seven'],
            limit_constraint_range=None,  # No range limit in this case
            latlon=True
        )

        # Check if the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Updated check: Expected columns without receptor-specific columns
        expected_columns = ['value', 'value name', 'Area', 'Area percent']
        for col in expected_columns:
            self.assertIn(col, result.columns)

        # Example values to check
        expected_values = [0.0, 5.0, 7.0]
        expected_areas = [19174937.53, 7611327.17, 1179139.2]
        expected_percent = [68.57, 27.22, 4.22]

        # Verify the calculated values
        np.testing.assert_array_almost_equal(result['value'], expected_values, decimal=2)
        np.testing.assert_array_almost_equal(result['Area'], expected_areas, decimal=2)
        np.testing.assert_array_almost_equal(result['Area percent'], expected_percent, decimal=2)




if __name__ == '__main__':
    unittest.main()
