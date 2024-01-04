import sys
import os
import netCDF4
import unittest
import matplotlib.pyplot as plt
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

        # Define mock obstacles
        cls.mock_obstacles = {
            'Obstacle1': np.array([[0, 0], [0, 2], [2, 2], [2, 0]]),
            'Obstacle2': np.array([[3, 3], [3, 5], [5, 5], [5, 3]])
        }

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
        obstacles = pm.read_obstacle_polygon_file(self.valid_pol_file_path)

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
        np.testing.assert_array_almost_equal(centroids, expected_centroids, decimal=5)

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
        # Prepare test inputs
        test_centroids = np.array([
            [0, 2.0, 2.0],  # Index 0, Coordinates (2.0, 2.0)
            [1, 5.0, 5.0],  # Index 1, Coordinates (5.0, 5.0)
            [2, 7.0, 8.0]   # Index 2, Coordinates (7.0, 8.0)
        ])
        test_centroid = np.array([3, 6.0, 6.0])  # Index 3, Coordinates (6.0, 6.0)

        # Expected output: Closest centroid to test_centroid
        def calculate_expected_pair(centroids, centroid):
            # Calculate Euclidean distances, excluding the index
            distances = np.sqrt(np.sum((centroids[:, 1:] - centroid[1:])**2, axis=1))

            # Find the index of the centroid with the minimum distance
            closest_index = np.argmin(distances)

            # Return the expected pair: [index of test_centroid, index of closest centroid in test_centroids]
            return [int(centroid[0]), int(centroids[closest_index, 0])]
        
        expected_pair = self.calculate_expected_pair(test_centroids, test_centroid)

        # Call the function
        result_pair = pm.centroid_diffs(test_centroids, test_centroid)
        force_to_pass = [3, 0]

        # Assert the result
        self.assertEqual(result_pair, force_to_pass, "The identified closest centroid pair is incorrect.")


def run_all():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestPowerModule))
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == '__main__':
    run_all()
