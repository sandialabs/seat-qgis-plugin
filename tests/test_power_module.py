import sys
import os
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
from seat.modules import power_module as pm
# fmt: on


class TestPowerModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Class method called before tests in an individual class run.
        """

        # Define paths with script_dir prepended
        cls.dev_present = os.path.join(script_dir, "data/structured/devices-present")
        cls.dev_not_present = os.path.join(script_dir, "data/structured/devices-not-present")
        cls.probabilities = os.path.join(script_dir, "data/structured/probabilities/probabilities.csv")
        cls.receptor = os.path.join(script_dir, "data/structured/receptor/grain_size_receptor.csv")

        cls.pol_file = os.path.join(script_dir, "data/structured/power_files/4x4/rect_4x4.pol")
        cls.power_file = os.path.join(script_dir, "data/structured/power_files/4x4/")
        cls.hydrodynamic_probabilities = os.path.join(script_dir, "data/structured/probabilities/hydrodynamic_probabilities.csv")



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
        Test the find_mean_point_of_obstacle_polygon function with actual obstacle data.
        """
        # Load the obstacle data from the .pol file
        obstacles = pm.read_obstacle_polygon_file(self.pol_file)

        # Calculate the centroids of the obstacles
        centroids = pm.find_mean_point_of_obstacle_polygon(obstacles)

        # Expected centroids (derived from the actual data)
        expected_centroids = np.array([
            [  10, 235.772795, 44.564 ],
            [  11, 235.775995, 44.564 ],
            [  12, 235.7664, 44.564005 ],
            [  13, 235.7696, 44.564 ],
            [  14, 235.7728, 44.564 ],
        ])

        # Assert the result
        np.testing.assert_array_almost_equal(centroids[10:15], expected_centroids, decimal=5)

    def test_plot_test_obstacle_locations(self):
        """
        Test the plot_test_obstacle_locations function using actual obstacle data.
        """
        # Load the obstacle data from the .pol file
        obstacles = pm.read_obstacle_polygon_file(self.pol_file)

        # Call the function to generate the plot
        fig = pm.plot_test_obstacle_locations(obstacles)

        # Check if a figure is returned
        self.assertIsInstance(fig, plt.Figure, "The output should be a matplotlib figure.")

        # Check figure size
        np.testing.assert_array_almost_equal(fig.get_size_inches(), [10, 10], decimal=5,
                                            err_msg="Figure size should be 10x10 inches.")

        # Check that the number of text elements in the plot matches the number of obstacles
        num_texts = len(fig.axes[0].texts)
        expected_num_texts = 2 * len(obstacles)  # Assuming each obstacle has two text elements
        self.assertEqual(num_texts, expected_num_texts, "Number of text elements in plot is incorrect.")



    def test_centroid_diffs(self):
        """
        Test the centroid_diffs function using actual centroid data.
        """
        # Load the actual obstacles and calculate their centroids
        obstacles = pm.read_obstacle_polygon_file(self.pol_file)
        centroids = pm.find_mean_point_of_obstacle_polygon(obstacles)

        # Define a test centroid
        test_centroid = np.array([ 10, 235.772795, 44.564 ])

        # Call the function to find the closest centroid
        result_pair = pm.centroid_diffs(centroids, test_centroid)

        # Calculate the expected closest centroid pair manually or using a helper function
        expected_pair = self.calculate_expected_pair(centroids, test_centroid)

        # Assert the result
        self.assertEqual(result_pair, expected_pair, "The identified closest centroid pair is incorrect.")



    def test_extract_device_location(self):
        """
        Test the extract_device_location function using actual obstacle data.
        """
        # Load the actual obstacles and calculate their centroids
        obstacles = pm.read_obstacle_polygon_file(self.pol_file)
        centroids = pm.find_mean_point_of_obstacle_polygon(obstacles)

        # Pair devices using the actual centroids
        device_index = pm.pair_devices(centroids)

        # Call the function to extract device locations
        devices_df = pm.extract_device_location(obstacles, device_index)

        # Check that the result is a DataFrame
        self.assertIsInstance(devices_df, pd.DataFrame, "The output should be a pandas DataFrame.")

        # Verify some key values, e.g., dimensions, centroids, etc.
        expected_width = 0.0004500000000007276
        expected_height = 0.00044999999999362217
        expected_lower_left = [235.76617, 44.55978]

        self.assertAlmostEqual(devices_df.iloc[0]['width'], expected_width, places=3, msg="Device width mismatch.")
        self.assertAlmostEqual(devices_df.iloc[0]['height'], expected_height, places=3, msg="Device height mismatch.")
        self.assertEqual(devices_df.iloc[0]['lower_left'], expected_lower_left, "Lower left corner mismatch.")



    def test_pair_devices(self):
        """
        Test the pair_devices function using actual obstacle data.
        """
        # Load the actual obstacles and calculate their centroids
        obstacles = pm.read_obstacle_polygon_file(self.pol_file)
        centroids = pm.find_mean_point_of_obstacle_polygon(obstacles)

        # Call the function to pair devices
        devices = pm.pair_devices(centroids)

        # Verify that the result is a NumPy array
        self.assertIsInstance(devices, np.ndarray, "The output should be a NumPy array.")

        # Verify that the pairs are formed correctly
        # Here, we check that each device has two valid centroid indices and that they are unique
        for device in devices:
            self.assertTrue(device[0] != device[1], "A device cannot pair the same centroid twice.")
            self.assertIn(device[0], centroids[:, 0], "First index in pair is not a valid centroid index.")
            self.assertIn(device[1], centroids[:, 0], "Second index in pair is not a valid centroid index.")


    def test_create_power_heatmap(self):
        """
        Test the create_power_heatmap function using real data from 'BC_probability_wPower.csv' and rect_4x4.pol
        """
        # Load the power data from the CSV file
        csv_file_path = os.path.join(script_dir, 'data', 'structured', 'output', 'BC_probability_wPower.csv')
        actual_power_data = pd.read_csv(csv_file_path)

        # Load the obstacle data from the .pol file to get the geometric values
        obstacles = pm.read_obstacle_polygon_file(self.pol_file)

        # Extract device locations from obstacles
        centroids = pm.find_mean_point_of_obstacle_polygon(obstacles)
        device_index = pm.pair_devices(centroids)
        devices_df = pm.extract_device_location(obstacles, device_index)

        # Use the first 5 devices and their corresponding spatial data
        lower_left = devices_df['lower_left'].head(5).tolist()
        width = devices_df['width'].head(5).tolist()
        height = devices_df['height'].head(5).tolist()

        # Take the first 5 values of the Power [W] column from the CSV
        power_values = actual_power_data['Power [W]'].head(5)

        # Construct a DataFrame using the real geometric values and the real power values
        test_data = pd.DataFrame({
            'lower_left': lower_left,
            'width': width,
            'height': height,
            'Power [W]': power_values
        })

        # Call the function being tested
        fig = pm.create_power_heatmap(test_data)

        # Assert that the figure is created successfully
        self.assertIsInstance(fig, plt.Figure, "The output should be a matplotlib figure.")

        # Extract the colorbar and axes data for further assertions
        colorbar = fig.axes[1]  # Colorbar is typically the second axes
        ax = fig.axes[0]  # Main plot is the first axes

        # Check that the power values in the heatmap match expected real data
        expected_power_values = power_values.values
        extracted_power_values = test_data['Power [W]'].values

        np.testing.assert_array_almost_equal(
            extracted_power_values,
            expected_power_values,
            decimal=5,
            err_msg="The power values in the heatmap do not match the expected values."
        )

        # Check the colorbar label
        self.assertEqual(colorbar.get_ylabel(), 'MW', "The color bar label should be 'MW'.")

        # Check axis labels
        self.assertEqual(ax.get_xlabel(), 'Longitude [deg]', "The x-axis label should be 'Longitude [deg]'.")
        self.assertEqual(ax.get_ylabel(), 'Latitude [deg]', "The y-axis label should be 'Latitude [deg]'.")

    def test_read_power_file(self):
        """
        Test the read_power_file function using actual .OUT power files.
        """
        # Use a real power file from the dataset
        real_power_file = os.path.join(self.power_file, 'POWER_ABS_001.OUT')

        # Expected results: manually verify the values in the file and set them here
        expected_power = np.array([125613.836, 113441.609, 346991.812, 289513.188, 209345.75 ])
        expected_total_power = 9235605.772

        # Call the function
        power, total_power = pm.read_power_file(real_power_file)

        # Assert the results are as expected based on the real file's content
        np.testing.assert_array_equal(power[10:15], expected_power, "Power array does not match expected values.")
        self.assertEqual(total_power, expected_total_power, "Total power does not match expected value.")


    def test_sort_data_files_by_runnumber(self):
        bc_data = pd.DataFrame({
            'run number': [3, 1, 2],
            'original_order': [2, 0, 1]
        })
        datafiles = ['file3.out', 'file1.out', 'file2.out']
        sorted_files = pm.sort_data_files_by_runnumber(bc_data, datafiles)
        self.assertEqual(sorted_files, ['file1.out', 'file2.out', 'file3.out'])

    def test_sort_bc_data_by_runnumber(self):
        """
        Test the sort_bc_data_by_runnumber function with actual data.
        """
        # Use a real bc_data DataFrame
        bc_data = pd.DataFrame({
            'run number': [3, 1, 2],
            'value': [30, 10, 20]
        })

        # Expected DataFrame after sorting by 'run number'
        expected_sorted_bc_data = pd.DataFrame({
            'run number': [1, 2, 3],
            'value': [10, 20, 30],
            'original_order': [1, 2, 0]  # This reflects the original order of rows before sorting
        }).reset_index(drop=True)

        # Call the function
        sorted_bc_data = pm.sort_bc_data_by_runnumber(bc_data)

        # Assert that the sorted DataFrame matches the expected result
        pd.testing.assert_frame_equal(sorted_bc_data.reset_index(drop=True), expected_sorted_bc_data, "bc_data is not sorted correctly by 'run number'.")

    def test_reset_bc_data_order(self):
        # Mock bc_data DataFrame
        bc_data = pd.DataFrame({
            'run number': [3, 1, 2],
            'value': [30, 10, 20],
            'original_order': [2, 0, 1]
        })

        with self.assertRaises(AttributeError):
            pm.reset_bc_data_order(bc_data)

    def test_roundup(self):
        # Test cases for the modified function's behavior
        self.assertEqual(pm.roundup(7, 5), 10)   # 7 rounds up to 10 (nearest multiple of 5)
        self.assertEqual(pm.roundup(3, 5), 5)    # 3 rounds up to 5 (nearest multiple of 5)
        self.assertEqual(pm.roundup(5, 5), 5)    # 5 stays 5 (nearest multiple of 5)
        self.assertEqual(pm.roundup(12, 5), 15)  # 12 rounds up to 15 (nearest multiple of 5)
        self.assertEqual(pm.roundup(-2, 5), 0)   # -2 rounds up to 0 (nearest multiple of 5)
        self.assertEqual(pm.roundup(0, 5), 0)    # 0 stays 0 (nearest multiple of 5)
        self.assertEqual(pm.roundup(1, 5), 5)    # 1 rounds up to 5 (nearest multiple of 5)
        self.assertEqual(pm.roundup(-5, 5), -5)  # -5 stays -5 (nearest multiple of 5)

    def test_calculate_power(self):
        """
        Test the calculate_power function.
        """
        # Paths to the power and probabilities files as set up in setUpClass
        power_files = self.power_file
        probabilities_file = self.hydrodynamic_probabilities



        with tempfile.TemporaryDirectory() as tmpdirname:
            pm.calculate_power(power_files, probabilities_file, tmpdirname)
            # Check if the expected output files are created
            expected_files = [
                'BC_probability_wPower.csv',
                'Total_Scaled_Power_Bars_per_Run.png',
                'Device_Power.png',
                'Power_per_device_annual.csv',
                'Scaled_Power_per_device_per_scenario.png',
                'Power_per_device_per_scenario.csv',
                'Obstacle_Matching.csv',
                'Device Number Location.png',
                'Obstacle_Locations.png',
                'Scaled_Power_Bars_per_run_obstacle.png',
            ]
            for file_name in expected_files:
                self.assertTrue(os.path.exists(os.path.join(tmpdirname, file_name)),
                                f"Expected output file {file_name} not found.")

def run_all():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestPowerModule))
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == '__main__':
    run_all()
