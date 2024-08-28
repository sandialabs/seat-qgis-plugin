import sys
import os
import netCDF4
import unittest
import numpy as np
import pandas as pd
from osgeo import gdal

from os.path import join

# Get the directory in which the current script is located
script_dir = os.path.dirname(os.path.realpath(__file__))

# Import seat
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# fmt: off
from seat.modules import shear_stress_module as ssm
# fmt: on

class TestShearStress(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Class method called before tests in an individual class run.
        """
        # Define paths with script_dir prepended
        cls.dev_present = join(script_dir, "data/structured/devices-present")
        cls.dev_not_present = join(script_dir, "data/structured/devices-not-present")
        cls.probabilities = join(script_dir, "data/structured/probabilities/hydrodynamic_probabilities.csv")
        cls.receptor_structured = join(script_dir, "data/structured/receptor/velocity_receptor.csv")

        # unstructured test cases
        cls.mec_present = join(script_dir, "data/unstructured/mec-present")
        cls.mec_not_present = join(script_dir, "data/unstructured/mec-not-present")
        cls.receptor_unstructured = join(script_dir, "data/unstructured/receptor/velocity_receptor.csv")

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
        Test the check_grid_define_vars function with both structured and unstructured datasets.

        This test checks if the function correctly identifies the type of grid (structured or unstructured),
        along with the names of the x-coordinate, y-coordinate, and shear stress variables in the dataset.
        """

        # Test structured data
        structured_data_paths = [
            join(self.dev_not_present, 'downsampled_devices_not_present_data.nc'),
            join(self.dev_present, 'downsampled_devices_present_data.nc')
        ]

        for path in structured_data_paths:
            with netCDF4.Dataset(path, 'r') as dataset:
                expected_gridtype = 'structured'
                expected_xvar = 'XZ'
                expected_yvar = 'YZ'
                expected_tauvar = 'TAUMAX'

                # Call the function with the actual dataset
                gridtype, xvar, yvar, tauvar = ssm.check_grid_define_vars(dataset)

                # Assert the function returns the expected values
                self.assertEqual(gridtype, expected_gridtype)
                self.assertEqual(xvar, expected_xvar)
                self.assertEqual(yvar, expected_yvar)
                self.assertEqual(tauvar, expected_tauvar)

        # Test unstructured data
        unstructured_data_paths = [
            join(self.mec_not_present, 'downsampled_0_tanana_1_map.nc'),
            join(self.mec_not_present, 'downsampled_0_tanana_100_map.nc'),
            join(self.mec_present, 'downsampled_9_tanana_1_map.nc'),
            join(self.mec_present, 'downsampled_9_tanana_100_map.nc')
        ]

        for path in unstructured_data_paths:
            with netCDF4.Dataset(path, 'r') as dataset:
                expected_gridtype = 'unstructured'
                expected_xvar = 'FlowElem_xcc'
                expected_yvar = 'FlowElem_ycc'
                expected_tauvar = 'taus'

                # Call the function with the actual dataset
                gridtype, xvar, yvar, tauvar = ssm.check_grid_define_vars(dataset)

                # Assert the function returns the expected values
                self.assertEqual(gridtype, expected_gridtype)
                self.assertEqual(xvar, expected_xvar)
                self.assertEqual(yvar, expected_yvar)
                self.assertEqual(tauvar, expected_tauvar)


    def test_calculate_shear_stress_stressors_structured(self):
        """
        Test the calculate_shear_stress_stressors function on structured data using means.
        """
        # Run the function
        dict_output, rx, ry, dx, dy, gridtype = ssm.calculate_shear_stress_stressors(
            self.dev_not_present,
            self.dev_present,
            self.probabilities,
            self.receptor_structured
        )

        # Define the expected means
        full_data_expected_means = {
            'shear_stress_without_devices': 3.664708133906911,
            'shear_stress_with_devices': 3.5320121139290905,
            'shear_stress_difference': -0.13269601997782032,
            'sediment_mobility_without_devices': 14762.226396577538,
            'sediment_mobility_with_devices': 14227.698511338009,
            'sediment_mobility_difference': -534.5278852395282,
            'sediment_mobility_classified': 1.0805218173639226,
            'sediment_grain_size': 0.04999999999999999,
            'shear_stress_risk_metric': -13688.74627406984
        }

        # Assert means are almost equal to the expected values
        for key, expected_mean in full_data_expected_means.items():
            actual_mean = np.nanmean(dict_output[key])
            self.assertAlmostEqual(actual_mean, expected_mean, places=3, msg=f"Mean mismatch for {key}")


        # Define expected values for spatial parameters
        expected_rx_first_5 = [0.0, 235.7164, 235.7164, 235.7164, 235.7164]
        expected_ry_first_5 = [0.0, 44.485, 44.489, 44.494, 44.5]
        expected_dx = 4.2132072
        expected_dy = 1.1755131
        expected_gridtype = 'structured'

        # Assert first values of rx, ry, and grid parameters
        np.testing.assert_array_almost_equal(rx.flatten()[:5], expected_rx_first_5, decimal=3)
        np.testing.assert_array_almost_equal(ry.flatten()[:5], expected_ry_first_5, decimal=3)
        self.assertAlmostEqual(dx, expected_dx)
        self.assertAlmostEqual(dy, expected_dy)
        self.assertEqual(gridtype, expected_gridtype)


    def test_calculate_shear_stress_stressors_unstructured(self):
        """
        Test the calculate_shear_stress_stressors function on unstructured data using means.
        """
        # Run the function
        dict_output, rx, ry, dx, dy, gridtype = ssm.calculate_shear_stress_stressors(
            self.mec_not_present,
            self.mec_present,
            probabilities_file='',
            receptor_filename=self.receptor_unstructured
        )

        # Expected means
        full_data_expected_means = {
            'shear_stress_without_devices': 1.6680394513408587,
            'shear_stress_with_devices': 1.6664031902991132,
            'shear_stress_difference': -0.001636261041400726,
            'sediment_mobility_without_devices': 6719.218862565925,
            'sediment_mobility_with_devices': 6712.627653915943,
            'sediment_mobility_difference': -6.591208645984725,
            'sediment_mobility_classified': -84.83865546218487,
            'sediment_grain_size': 0.05,
            'shear_stress_risk_metric': -2035.757266273858
        }
        expected_rx = 399974.68467022
        expected_ry = 7160695.08352798
        expected_dx = 37.78892640129551
        expected_dy = 37.78892640129551
        expected_gridtype = 'unstructured'

        self.assertIsInstance(dict_output, dict)
        self.assertIsInstance(rx, np.ndarray)
        self.assertIsInstance(ry, np.ndarray)
        self.assertTrue(isinstance(dx, float) or isinstance(dx, np.floating))
        self.assertTrue(isinstance(dy, float) or isinstance(dy, np.floating))
        self.assertIsInstance(gridtype, str)

        # Calculate and assert means
        for key, expected_mean in full_data_expected_means.items():
            actual_mean = np.nanmean(dict_output[key])
            self.assertAlmostEqual(actual_mean, expected_mean, places=4, msg=f"Mean mismatch for {key}")

        # Assert spatial parameters
        np.testing.assert_array_almost_equal(rx[0, 0], expected_rx)
        np.testing.assert_array_almost_equal(ry[0, 0], expected_ry)
        self.assertAlmostEqual(dx, expected_dx)
        self.assertAlmostEqual(dy, expected_dy)
        self.assertEqual(gridtype, expected_gridtype)



    def test_run_shear_stress_stressor_structured(self):
        output_path = "test_output_structured"  # Define a directory for test outputs
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Call the function with test data
        result = ssm.run_shear_stress_stressor(
            self.dev_present,
            self.dev_not_present,
            self.probabilities,
            crs=4326,
            output_path=output_path,
            receptor_filename=self.receptor_structured,
            secondary_constraint_filename=None
        )

        expected_keys = [
            'shear_stress_without_devices', 'shear_stress_with_devices', 'shear_stress_difference',
            'sediment_mobility_without_devices', 'sediment_mobility_with_devices', 'sediment_mobility_difference',
            'sediment_mobility_classified', 'sediment_grain_size', 'shear_stress_risk_metric'
        ]

        expected_means = {
            'shear_stress_without_devices': 3.664708375930786,
            'shear_stress_with_devices': 3.5320122241973877,
            'shear_stress_difference': -0.1326960176229477,
            'sediment_mobility_without_devices': 14762.2265625,
            'sediment_mobility_with_devices': 14227.6982421875,
            'sediment_mobility_difference': -534.5278930664062,
            'sediment_mobility_classified': 1.080521821975708,
            'sediment_grain_size': 0.05000000447034836,
            'shear_stress_risk_metric': -13688.7470703125
        }

        self.assertIsInstance(result, dict)
        for key in expected_keys:
            self.assertIn(key, result)

            file_path = result[key]
            self.assertTrue(os.path.isfile(file_path), f"GeoTIFF file '{file_path}' was not created.")

            # Validate that the file is a valid GeoTIFF using GDAL
            dataset = gdal.Open(file_path)
            self.assertIsNotNone(dataset, f"Failed to open file '{file_path}' with GDAL.")
            self.assertEqual(dataset.RasterCount, 1, f"GeoTIFF '{file_path}' should have exactly one band.")
            self.assertEqual(dataset.GetDriver().ShortName, 'GTiff', f"File '{file_path}' is not a valid GeoTIFF.")

            # Read the data and calculate the mean
            band = dataset.GetRasterBand(1)
            data = band.ReadAsArray()
            mean_value = np.nanmean(data)

            self.assertAlmostEqual(mean_value, expected_means[key], places=5, msg=f"Mean mismatch for {key}")

            dataset = None  # Close the dataset

        # Clean up test output files
        for file_path in result.values():
            if os.path.exists(file_path):
                os.remove(file_path)

        for file in os.listdir(output_path):
            file_path = os.path.join(output_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        os.rmdir(output_path)


    def test_run_shear_stress_stressor_unstructured(self):
        """
        Test the run_shear_stress_stressor function to ensure it correctly processes unstructured input data,
        generates the expected GeoTIFFs and area change statistics files, and print data values for inspection.
        """

        output_path = "test_output_unstructured"  # Define a directory for test outputs
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Call the function with unstructured test data
        result = ssm.run_shear_stress_stressor(
            self.mec_present,
            self.mec_not_present,
            probabilities_file='',
            crs=4326,
            output_path=output_path,
            receptor_filename=self.receptor_unstructured,
            secondary_constraint_filename=None
        )

        # Verify that the function returns a dictionary with the expected keys
        expected_keys = [
            'shear_stress_without_devices', 'shear_stress_with_devices', 'shear_stress_difference',
            'sediment_mobility_without_devices', 'sediment_mobility_with_devices', 'sediment_mobility_difference',
            'sediment_mobility_classified', 'sediment_grain_size', 'shear_stress_risk_metric'
        ]

        expected_means = {
            'shear_stress_without_devices': 1.6680394411087036,
            'shear_stress_with_devices': 1.66640305519104,
            'shear_stress_difference': -0.001636261004023254,
            'sediment_mobility_without_devices': 6719.21923828125,
            'sediment_mobility_with_devices': 6712.6279296875,
            'sediment_mobility_difference': -6.591207981109619,
            'sediment_mobility_classified': -84.83865356445312,
            'sediment_grain_size': 0.04999999701976776,
            'shear_stress_risk_metric': -2035.75732421875
        }

        self.assertIsInstance(result, dict)
        for key in expected_keys:
            self.assertIn(key, result)

            file_path = result[key]
            self.assertTrue(os.path.isfile(file_path), f"GeoTIFF file '{file_path}' was not created.")

            # Validate that the file is a valid GeoTIFF using GDAL
            dataset = gdal.Open(file_path)
            self.assertIsNotNone(dataset, f"Failed to open file '{file_path}' with GDAL.")
            self.assertEqual(dataset.RasterCount, 1, f"GeoTIFF '{file_path}' should have exactly one band.")
            self.assertEqual(dataset.GetDriver().ShortName, 'GTiff', f"File '{file_path}' is not a valid GeoTIFF.")

            # Read the data and calculate the mean
            band = dataset.GetRasterBand(1)
            data = band.ReadAsArray()
            mean_value = np.nanmean(data)

            self.assertAlmostEqual(mean_value, expected_means[key], places=5, msg=f"Mean mismatch for {key}")

            dataset = None  # Close the dataset

        # Clean up test output files
        for file_path in result.values():
            if os.path.exists(file_path):
                os.remove(file_path)

        for file in os.listdir(output_path):
            file_path = os.path.join(output_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        os.rmdir(output_path)




def run_all():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestShearStress))
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == '__main__':
    run_all()
