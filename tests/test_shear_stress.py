import sys
import os
import netCDF4
import unittest
import numpy as np
from qgis.core import QgsApplication

from os.path import join

# Get the directory in which the current script is located
script_dir = os.path.dirname(os.path.realpath(__file__))

# Import seat
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# fmt: off
from seat.modules import shear_stress_module as ssm
# fmt: on


# Mock Interface
class MockIface:
    pass


class TestShearStress(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Class method called before tests in an individual class run.
        """
        # Set up the mock netCDF file
        cls.mock_netcdf_data = 'mock_netcdf.nc'
        cls.create_mock_netcdf(cls.mock_netcdf_data)

        # Define paths with script_dir prepended
        cls.dev_present = join(script_dir, "data/structured/devices-present")
        cls.dev_not_present = join(script_dir, "data/structured/devices-not-present")
        cls.probabilities = join(script_dir, "data/structured/probabilities/probabilities.csv")
        cls.receptor = join(script_dir, "data/structured/receptor/grain_size_receptor.csv")

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

    def test_critical_shear_stress(self):
        """
        Test the critical_shear_stress function with a set of predefined grain sizes.

        This test checks if the critical shear stress values calculated by the
        critical_shear_stress function match the expected values for a given set of grain sizes.
        The expected values are predetermined and the test checks for a close match with 
        a specific decimal precision.

        Parameters:
        D_meters (np.array): An array of grain sizes in meters.
        expected_output (np.array): An array of expected critical shear stress values.
        """
        D_meters = np.array([0.001, 0.002])
        expected_output = np.array([0.52054529, 1.32156154])
        result = ssm.critical_shear_stress(D_meters)
        np.testing.assert_almost_equal(result, expected_output, decimal=5)

    def test_classify_mobility(self):
        """
        Test the classify_mobility function with predefined mobility parameters.

        This test verifies the correct classification of sediment mobility based on the
        mobility parameters with and without device runs. It checks if the function correctly
        classifies each case as new erosion, increased erosion, reduced erosion, no change,
        reduced deposition, increased deposition, or new deposition.

        Parameters:
        mobility_parameter_dev (np.array): Array of mobility parameters for with device runs.
        mobility_parameter_nodev (np.array): Array of mobility parameters for without (baseline) device runs.
        expected_classification (np.array): Array of expected numerical classifications.
        """
        mobility_parameter_dev = np.array([
            1.0,  # New Erosion (dev >= 1, nodev < 1)
            1.2,  # Increased Erosion
            1.0,  # Reduced Erosion
            1.0,   # No Change
            0.6,  # Reduced Deposition (dev > nodev, both < 1)
            0.2,  # Increased Deposition (dev < nodev, both < 1)
            0.6,  # New Deposition (dev < 1, nodev >= 1)
        ])

        # Baseline (no device) run parameters
        mobility_parameter_nodev = np.array([
            0.9,  # New Erosion
            1.0,  # Increased Erosion
            1.1,  # Reduced Erosion
            1.0,   # No Change
            0.5,  # Reduced Deposition
            0.3,  # Increased Deposition
            1.0,  # New Deposition
        ])
        expected_classification = np.array([3.,  2.,  1., 0., -1., -2.,  -3.])
        result = ssm.classify_mobility(
            mobility_parameter_dev, mobility_parameter_nodev)
        np.testing.assert_array_equal(result, expected_classification)

    def test_check_grid_define_vars(self):
        """
        Test the check_grid_define_vars function with a mock dataset.

        This test checks if the function correctly identifies the type of grid (structured or unstructured),
        along with the names of the x-coordinate, y-coordinate, and shear stress variables in the dataset.
        """
        # Open and use the mock netCDF dataset
        with netCDF4.Dataset(self.mock_netcdf_data, 'r') as mock_dataset:
            expected_gridtype = 'structured'
            expected_xvar = 'x_coord'
            expected_yvar = 'y_coord'
            expected_tauvar = 'TAUMAX'

            # Call the function with the mock dataset
            gridtype, xvar, yvar, tauvar = ssm.check_grid_define_vars(
                mock_dataset)

            # Assert the function returns the expected values
            self.assertEqual(gridtype, expected_gridtype)
            self.assertEqual(xvar, expected_xvar)
            self.assertEqual(yvar, expected_yvar)
            self.assertEqual(tauvar, expected_tauvar)

    def test_calculate_shear_stress_stressors(self):
        """
        Test the calculate_shear_stress_stressors function.
        """
        dict_output, rx, ry, dx, dy, gridtype = ssm.calculate_shear_stress_stressors(
            self.dev_not_present,
            self.dev_present,
            self.probabilities,
        )

        self.assertIsInstance(dict_output, dict)
        self.assertIsInstance(rx, np.ndarray)
        self.assertIsInstance(ry, np.ndarray)
        self.assertTrue(isinstance(dx, float) or isinstance(dx, np.floating))
        self.assertTrue(isinstance(dy, float) or isinstance(dy, np.floating))
        self.assertIsInstance(gridtype, str)


    def test_run_shear_stress_stressor(self):
            """
            Test the run_shear_stress_stressor function to ensure it correctly processes input data
            and generates the expected geotiffs and area change statistics files.
            """
            output_path = "test_output"  # Define a directory for test outputs
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            # Call the function with test data
            result = ssm.run_shear_stress_stressor(
                self.dev_present,
                self.dev_not_present,
                self.probabilities,
                crs=4326,  
                output_path=output_path,
                receptor_filename=self.receptor,
                secondary_constraint_filename=None
            )

            # Verify that the function returns a dictionary with the expected keys
            expected_keys = [
            'shear_stress_without_devices', 'shear_stress_with_devices', 'shear_stress_difference',
            'sediment_mobility_without_devices', 'sediment_mobility_with_devices', 'sediment_mobility_difference',
            'sediment_mobility_classified', 'sediment_grain_size', 'shear_stress_risk_metric'
        ]
            self.assertIsInstance(result, dict)
            for key in expected_keys:
                self.assertIn(key, result)
                self.assertTrue(os.path.isfile(result[key]))

            # Clean up test output files
            for file_path in result.values():
                if os.path.exists(file_path):
                    os.remove(file_path)

            # Remove any additional files in the directory
            for file in os.listdir(output_path):
                file_path = os.path.join(output_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            # Now remove the directory
            os.rmdir(output_path)

def run_all():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestShearStress))
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == '__main__':
    run_all()
