import sys
import os
import netCDF4
import unittest
import numpy as np
from qgis.core import QgsApplication


# Get the directory in which the current script is located
script_dir = os.path.dirname(os.path.realpath(__file__))

# Import seat
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# fmt: off
from seat import shear_stress_module as ssm
from seat import power_module as pm
# fmt: on


# Mock Interface
class MockIface:
    pass


class TestPowerModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Class method called before tests in an individual class run.
        """
        # Set up the mock netCDF file
        cls.mock_netcdf_data = 'mock_netcdf.nc'
        cls.create_mock_netcdf(cls.mock_netcdf_data)

        # Define paths with script_dir prepended
        cls.dev_present = os.path.join(script_dir, "data/structured/devices-present")
        cls.dev_not_present = os.path.join(script_dir, "data/structured/devices-not-present")
        cls.probabilities = os.path.join(script_dir, "data/structured/probabilities/probabilities.csv")
        cls.receptor = os.path.join(script_dir, "data/structured/receptor/grain_size_receptor.csv")

        cls.valid_pol_file_path = os.path.join(script_dir, "data/structured/power_files/4x4/rect_4x4.pol")


    @classmethod
    def tearDownClass(cls):
        """
        Class method called after tests in an individual class are run.
        """
        # Clean up the mock netCDF file
        if os.path.exists(cls.mock_netcdf_data):
            os.remove(cls.mock_netcdf_data)

    @staticmethod
    def create_mock_netcdf(filename):
        """
        Create a mock netCDF file with predefined structure and variables.
        The structure is tailored to test the check_grid_define_vars function.
        """
        with netCDF4.Dataset(filename, "w", format="NETCDF4") as dataset:
            # Create dimensions
            dataset.createDimension('x', None)
            dataset.createDimension('y', None)

            # Create coordinate variables
            x = dataset.createVariable('x_coord', np.float32, ('x',))
            y = dataset.createVariable('y_coord', np.float32, ('y',))

            # Create a variable with coordinates attribute
            taumax = dataset.createVariable('TAUMAX', np.float32, ('x', 'y'))
            taumax.coordinates = 'x_coord y_coord'

            # Add some data to the variables
            x[:] = np.arange(0, 10, 1)
            y[:] = np.arange(0, 20, 1)
            taumax[:, :] = np.random.rand(10, 20)

        return filename


    def test_valid_pol_file(self):
        """
        Test the read_obstacle_polygon_file function with a valid .pol file.
        """
        # Call the function with the valid .pol file
        obstacles = pm.read_obstacle_polygon_file(self.valid_pol_file_path)

        self.assertIsInstance(obstacles, dict, "The output should be a dictionary.")

    def test_invalid_file_path(self):
        """
        Test that a FileNotFoundError is raised for an invalid file path.
        """
        bad_path = os.path.join(script_dir, "non_existent_directory/non_existent_file.pol")
        with self.assertRaises(FileNotFoundError):
            pm.read_obstacle_polygon_file(bad_path)


def run_all():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestPowerModule))
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == '__main__':
    run_all()
