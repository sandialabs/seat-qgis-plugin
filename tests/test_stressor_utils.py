import sys
import os
import netCDF4
import unittest
import numpy as np
import pandas as pd
from qgis.core import QgsApplication
from osgeo import gdal, osr
from unittest.mock import patch, MagicMock

# Get the directory in which the current script is located
script_dir = os.path.dirname(os.path.realpath(__file__))

# Import seat
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# fmt: off
from seat import stressor_utils as su

# fmt: on
# from seat.stressor_utils import estimate_grid_spacing

class TestEstimateGridSpacing(unittest.TestCase):

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


class TestCreateStructuredArrayFromUnstructured(unittest.TestCase):

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


class TestRedefineStructuredGrid(unittest.TestCase):

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

class TestResampleStructuredGrid(unittest.TestCase):

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

#TODO: Add tests for calc_receptor_array
class TestCalcReceptorArray(unittest.TestCase):
    pass
    # @patch('seat.stressor_utils.gdal.Open')
    # def test_with_tif_file(self, mock_gdal_open):
    #     # Mock the behavior of gdal.Open for a .tif file
    #     mock_gdal_open.return_value = MagicMock()
    #     # ... Set up the necessary attributes and return values for the mock object ...
        
    #     x = np.array([0, 1, 2])
    #     y = np.array([0, 1, 2])

    #     receptor_array = su.calc_receptor_array('test.tif', x, y)

        # Assert the output
        # ... Add your assertions here ...

    # @patch('seat.stressor_utils.pd.read_csv')
    # def test_with_csv_file(self, mock_read_csv):
    #     # Mock the behavior of pd.read_csv for a .csv file
    #     mock_read_csv.return_value = pd.DataFrame([[1]])

    #     x = np.array([0, 1, 2])
    #     y = np.array([0, 1, 2])

    #     receptor_array = su.calc_receptor_array('test.csv', x, y)

    #     # Assert the output
    #     # ... Add your assertions here ...

    # def test_without_file(self):
    #     x = np.array([0, 1, 2])
    #     y = np.array([0, 1, 2])

    #     receptor_array = su.calc_receptor_array('', x, y)

    #     # Assert the output
    #     expected_array = 200e-6 * np.ones(x.shape)
    #     np.testing.assert_array_equal(receptor_array, expected_array)

class TestTrimZeros(unittest.TestCase):

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


class TestCreateRaster(unittest.TestCase):

    @patch('seat.stressor_utils.gdal.GetDriverByName')
    def test_create_raster(self, mock_get_driver_by_name):
        # Mock the GDAL driver and its Create method
        mock_driver = MagicMock()
        mock_create = MagicMock()
        mock_driver.Create = mock_create
        mock_get_driver_by_name.return_value = mock_driver

        # Call the function
        output_path = 'test_output.tif'
        cols = 100
        rows = 100
        nbands = 1
        eType = gdal.GDT_Float32

        raster = su.create_raster(output_path, cols, rows, nbands, eType)

        # Assert that GetDriverByName and Create were called correctly
        mock_get_driver_by_name.assert_called_once_with("GTiff")
        mock_create.assert_called_once_with(output_path, cols, rows, nbands, eType=gdal.GDT_Float32)

        # Assert that the function returns a mock raster object
        self.assertEqual(raster, mock_create.return_value)

class TestNumpyArrayToRaster(unittest.TestCase):

    @patch('seat.stressor_utils.os.path.exists')
    @patch('seat.stressor_utils.gdal')
    @patch('seat.stressor_utils.osr.SpatialReference')
    def test_numpy_array_to_raster(self, mock_spatial_reference, mock_gdal, mock_os_path_exists):
        # Mock the behavior of GDAL and os.path.exists
        mock_os_path_exists.return_value = True
        mock_output_raster = MagicMock()
        mock_band = MagicMock()
        mock_output_raster.GetRasterBand.return_value = mock_band
        mock_spatial_ref_instance = mock_spatial_reference.return_value
        mock_spatial_ref_instance.ExportToWkt.return_value = 'WKT'

        # Test data
        numpy_array = np.array([[1, 2], [3, 4]])
        bounds = [0, 0]
        cell_resolution = [1, 1]
        spatial_reference_system_wkid = 4326
        output_path = 'test_output.tif'

        result_path = su.numpy_array_to_raster(mock_output_raster, numpy_array, bounds, cell_resolution, spatial_reference_system_wkid, output_path)

        # Asserts
        mock_output_raster.SetProjection.assert_called_once()
        mock_output_raster.SetGeoTransform.assert_called_once()
        mock_band.WriteArray.assert_called_once_with(numpy_array)
        mock_band.FlushCache.assert_called_once()
        mock_band.ComputeStatistics.assert_called_once_with(False)
        self.assertEqual(result_path, output_path)

class TestFindUtmSrid(unittest.TestCase):

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

# TODO: Add tests for read_raster
class TestReadRaster(unittest.TestCase):
    pass
    # @patch('seat.stressor_utils.gdal.Open')
    # def test_read_raster(self, mock_gdal_open):
    #     # Mock the behavior of gdal.Open and necessary methods
    #     mock_data = MagicMock()
    #     mock_band = MagicMock()
    #     mock_data.GetRasterBand.return_value = mock_band
    #     mock_band.ReadAsArray.return_value = np.array([[1, 2], [3, 4]])
    #     mock_data.GetGeoTransform.return_value = (0, 1, 0, 0, 0, -1)
    #     mock_data.RasterXSize = 2
    #     mock_data.RasterYSize = 2
    #     mock_gdal_open.return_value = mock_data

    #     # Call the function
    #     raster_name = 'test_raster.tif'
    #     rx, ry, raster_array = su.read_raster(raster_name)

    #     # Expected values
    #     expected_rx = np.array([[0.5, 1.5], [0.5, 1.5]])
    #     expected_ry = np.array([[0.5, 0.5], [-0.5, -0.5]])
    #     expected_raster_array = np.array([[1, 2], [3, 4]])

    #     # Asserts
    #     np.testing.assert_array_equal(rx, expected_rx)
    #     np.testing.assert_array_equal(ry, expected_ry)
    #     np.testing.assert_array_equal(raster_array, expected_raster_array)

#TODO: Add tests for secondary_constraint_geotiff_to_numpy
class TestSecondaryConstraintGeotiffToNumpy(unittest.TestCase):
    pass
    # @patch('seat.stressor_utils.gdal.Open')
    # def test_secondary_constraint_geotiff_to_numpy(self, mock_gdal_open):
    #     # Mock the behavior of gdal.Open and necessary methods
    #     mock_data = MagicMock()
    #     mock_band = MagicMock()
    #     mock_data.GetRasterBand.return_value = mock_band
    #     mock_band.ReadAsArray.return_value = np.array([[1, 2], [3, 4]])
    #     mock_data.GetGeoTransform.return_value = (0, 1, 0, 0, 0, -1)
    #     mock_data.RasterXSize = 2
    #     mock_data.RasterYSize = 2
    #     # Use a valid WKT string (e.g., WGS 84)
    #     mock_data.GetProjection.return_value = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]'
    #     mock_gdal_open.return_value = mock_data

    #     # Call the function
    #     filename = 'test_raster.tif'
    #     x_grid, y_grid, array = su.secondary_constraint_geotiff_to_numpy(filename)

    #     # Expected values
    #     expected_x_grid = np.array([[0.5, 1.5], [0.5, 1.5]])
    #     expected_y_grid = np.array([[0.5, 0.5], [-0.5, -0.5]])
    #     expected_array = np.array([[1, 2], [3, 4]])

    #     # Asserts
    #     np.testing.assert_array_equal(x_grid, expected_x_grid)
    #     np.testing.assert_array_equal(y_grid, expected_y_grid)
    #     np.testing.assert_array_equal(array, expected_array)

class TestCalculateCellArea(unittest.TestCase):

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

if __name__ == '__main__':
    unittest.main()
