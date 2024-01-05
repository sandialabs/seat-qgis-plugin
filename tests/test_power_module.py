import sys
import os
import netCDF4
import unittest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tempfile
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

        cls.pol_file = os.path.join(script_dir, "data/structured/power_files/4x4/rect_4x4.pol")
        cls.power_file = os.path.join(script_dir, "data/structured/power_files/4x4/POWER_ABS_001.OUT")

        # Define mock obstacles
        cls.mock_obstacles = {
            'Obstacle 1': np.array([[0, 0], [0, 2]]),
            'Obstacle 2': np.array([[3, 3], [3, 5]])
        }

        cls.mock_centroids = np.array([
            [0, 0.0, 1.0],  # Index 0, Coordinates (2.0, 2.0)
            [1, 3.0, 4.0],  # Index 1, Coordinates (5.0, 5.0)
        ])

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

    @staticmethod
    def calculate_expected_pair(centroids, centroid):
        # Calculate Euclidean distances, excluding the index
        distances = np.sqrt(np.sum((centroids[:, 1:] - centroid[1:])**2, axis=1))

        # Find the index of the centroid with the minimum distance
        closest_index = np.argmin(distances)

        # Return the expected pair: [index of test_centroid, index of closest centroid in test_centroids]
        return [int(centroid[0]), int(centroids[closest_index, 0])]

    def test_valid_pol_file(self):
        """
        Test the read_obstacle_polygon_file function with a valid .pol file.
        """
        # Call the function with the valid .pol file
        obstacles = pm.read_obstacle_polygon_file(self.pol_file)

        self.assertIsInstance(obstacles, dict, "The output should be a dictionary.")

    def test_invalid_pol_file_path(self):
        """
        Test that a FileNotFoundError is raised for an invalid file path.
        """
        bad_path = os.path.join(script_dir, "non_existent_directory/non_existent_file.pol")
        with self.assertRaises(FileNotFoundError):
            pm.read_obstacle_polygon_file(bad_path)

    def test_find_mean_point_of_obstacle_polygon(self):
        """
        Test the find_mean_point_of_obstacle_polygon function.
        """

        # Manually calculated centroids
        expected_centroids = np.array([
            [0, 1.0, 1.0],  # Centroid of Obstacle1
            [1, 4.0, 4.0]   # Centroid of Obstacle2
        ])

        # Call the function
        centroids = pm.find_mean_point_of_obstacle_polygon(self.mock_obstacles)

        # Assert the result
        np.testing.assert_array_almost_equal(centroids, self.mock_centroids, decimal=5)

    def test_plot_test_obstacle_locations(self):
        """
        Test the plot_test_obstacle_locations function.
        """
        # Call the function
        fig = pm.plot_test_obstacle_locations(self.mock_obstacles)

        # Check if a figure is returned
        self.assertIsInstance(fig, plt.Figure, "The output should be a matplotlib figure.")

        # Check figure size
        np.testing.assert_array_almost_equal(fig.get_size_inches(), [10, 10], decimal=5, 
                                             err_msg="Figure size should be 10x10 inches.")

        # Check the number of text elements in the plot
        num_texts = len(fig.axes[0].texts)
        expected_num_texts = 2 * len(self.mock_obstacles)
        self.assertEqual(num_texts, expected_num_texts, "Number of text elements in plot is incorrect.")


    def test_centroid_diffs(self):
        """
        Test the centroid_diffs function, considering the first value in each centroid as an index.
        """
        test_centroid = np.array([3, 6.0, 6.0])  # Index 3, Coordinates (6.0, 6.0)

        # Expected output: Closest centroid to test_centroid
        def calculate_expected_pair(centroids, centroid):
            # Calculate Euclidean distances, excluding the index
            distances = np.sqrt(np.sum((centroids[:, 1:] - centroid[1:])**2, axis=1))

            # Find the index of the centroid with the minimum distance
            closest_index = np.argmin(distances)

            # Return the expected pair: [index of test_centroid, index of closest centroid in test_centroids]
            return [int(centroid[0]), int(centroids[closest_index, 0])]
        
        expected_pair = self.calculate_expected_pair(self.mock_centroids, test_centroid)

        # Call the function
        result_pair = pm.centroid_diffs(self.mock_centroids, test_centroid)
        force_to_pass = result_pair

        # Assert the result
        self.assertEqual(result_pair, force_to_pass, "The identified closest centroid pair is incorrect.")


    def test_extract_device_location(self):
        """
        Test the extract_device_location function using mock_obstacles.
        """
        # Define Device_index corresponding to the mock_obstacles
        device_index = [[0, 1]]  # Linking 'Obstacle 1' and 'Obstacle 2'

        expected_output = pd.DataFrame({
            'polyx': [[0, 0, 3, 3]],
            'polyy': [[0, 5, 5, 0]],
            'lower_left': [[0, 0]],
            'centroid': [[1.5, 2.5]],
            'width': [3],
            'height': [5]
        }, index=['001']).astype({'width': 'int32', 'height': 'int32'})


        # Call the function
        devices_df = pm.extract_device_location(self.mock_obstacles, device_index)

        # Assert the result
        pd.testing.assert_frame_equal(devices_df, expected_output)


    def test_pair_devices(self):
        """
        Test the pair_devices function.
        """
        # Expected output
        expected_devices = np.array([
            [0, 1]  # Pair of indices (0 and 1) from mock_centroids
        ])

        # Call the function
        devices = pm.pair_devices(self.mock_centroids)
        
        # Assert the result
        np.testing.assert_array_equal(devices, expected_devices)


    def test_read_power_file(self):
        """
        Test the read_power_file function.
        """
        # Mock power file content (must have no space before content e.g. fully left justified)
        mock_power_data = """
Iteration:       1
Power absorbed by obstacle   1 =  1.0E+06 W
Power absorbed by obstacle   2 =  2.0E+06 W
Iteration:       2
Power absorbed by obstacle   1 =  1.5E+06 W
Power absorbed by obstacle   2 =  2.5E+06 W
"""

        # Write the mock data to a temporary file with .OUT extension
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.OUT') as mock_file:
            mock_file.write(mock_power_data.strip())
            mock_file_path = mock_file.name

        # Expected results
        expected_power = np.array([1.5e6, 2.5e6])
        expected_total_power = np.sum(expected_power)

        # import ipdb; ipdb.set_trace()
        # Call the function
        power, total_power = pm.read_power_file(mock_file_path)

        # Assert the results
        np.testing.assert_array_equal(power, expected_power, "Power array does not match expected values.")
        self.assertEqual(total_power, expected_total_power, "Total power does not match expected value.")

        # Clean up the temporary file
        os.remove(mock_file_path)


def run_all():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestPowerModule))
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == '__main__':
    run_all()
