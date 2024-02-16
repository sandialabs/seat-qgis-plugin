import sys
import os
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, PropertyMock
import netCDF4

# Get the directory in which the current script is located
script_dir = os.path.dirname(os.path.realpath(__file__))

# Import seat
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# fmt: off
from seat import velocity_module as vm

# fmt: on
# from seat.stressor_utils import estimate_grid_spacing

class TestClassifyMotility(unittest.TestCase):
    def test_classify_motility(self):
        # Define test cases
        test_cases = [
            {
                "name": "Motility Stops",
                "dev": np.array([0.5, 0.9]),  # motility with device
                "nodev": np.array([1.5, 1.1]),  # motility without device
                "expected": np.array([-1, -1]),  # expected classification
            },
            {
                "name": "Reduced Motility",
                "dev": np.array([1.2, 1.0]), 
                "nodev": np.array([2.5, 2.1]),
                "expected": np.array([1, 1]),
            },
            {
                "name": "Increased Motility",
                "dev": np.array([3.0, 2.6]),
                "nodev": np.array([1.0, 1.5]),
                "expected": np.array([2, 2]),
            },
            {
                "name": "New Motility",
                "dev": np.array([2.0, 1.8]),
                "nodev": np.array([0.5, 0.9]),
                "expected": np.array([3, 3]),
            },
            {
                "name": "No Change",
                "dev": np.array([1.0, 1.0]),
                "nodev": np.array([1.0, 1.0]),
                "expected": np.array([0, 0]),
            },
        ]

        for case in test_cases:
            with self.subTest(name=case["name"]):
                # Call the classify_motility function
                classification = vm.classify_motility(case["dev"], case["nodev"])

                # Assert the expected result
                np.testing.assert_array_equal(classification, case["expected"], err_msg=f"Failed on case: {case['name']}")


class TestCheckGridDefineVars(unittest.TestCase):
    def test_structured_grid_with_coordinates(self):
        # Mock dataset for structured grid
        dataset = MagicMock()
        dataset.variables = {
            'U1': MagicMock(coordinates='X1 Y1'),
            'V1': MagicMock()
        }

        # Expected results
        expected = ('structured', 'X1', 'Y1', 'U1', 'V1')

        # Call function
        result = vm.check_grid_define_vars(dataset)

        # Assert
        self.assertEqual(result, expected)

    def test_structured_grid_without_coordinates_fallback(self):
        # Mock dataset for structured grid without coordinates in U1 variable
        dataset = MagicMock()
        dataset.variables = {
            'U1': MagicMock(),
            'V1': MagicMock()
        }
        # Mock the absence of 'coordinates' attribute by throwing an AttributeError
        type(dataset.variables['U1']).coordinates = property(lambda _: exec('raise AttributeError'))

        # Expected results
        expected = ('structured', 'XCOR', 'YCOR', 'U1', 'V1')

        # Call function
        result = vm.check_grid_define_vars(dataset)

        # Assert
        self.assertEqual(result, expected)

    def test_unstructured_grid(self):
        # Mock dataset for unstructured grid
        dataset = MagicMock()
        dataset.variables = {
            'ucxa': MagicMock(coordinates='X2 Y2'),
            'ucya': MagicMock()
        }

        # Expected results
        expected = ('unstructured', 'X2', 'Y2', 'ucxa', 'ucya')

        # Call function
        result = vm.check_grid_define_vars(dataset)

        # Assert
        self.assertEqual(result, expected)

class TestCalculateVelocityStressors(unittest.TestCase):
    @patch('os.listdir')
    @patch('netCDF4.Dataset') 
    def test_calculate_velocity_stressors(self, mock_dataset, mock_listdir):
        # Setup mock for listdir to simulate finding specific .nc files
        mock_listdir.side_effect = lambda x: ['last_2_runs.nc'] if 'fpath_dev' in x else ['last_2_runs.nc']

        # Setup MagicMock for the netCDF dataset
        mock_u_data = np.random.rand(5, 5)  # Random data for demonstration
        mock_v_data = np.random.rand(5, 5)
        mock_x_data = np.arange(5)
        mock_y_data = np.arange(5)

        mock_uvar = MagicMock()
        mock_uvar.data = mock_u_data
        mock_vvar = MagicMock()
        mock_vvar.data = mock_v_data
        mock_xvar = MagicMock()
        mock_xvar.data = mock_x_data
        mock_yvar = MagicMock()
        mock_yvar.data = mock_y_data

        mock_ds_instance = MagicMock()
        mock_ds_instance.variables = {
            'U1': mock_uvar,
            'V1': mock_vvar,
            'X1': mock_xvar,
            'Y1': mock_yvar
        }
        mock_dataset.return_value = mock_ds_instance

        # Define inputs
        fpath_nodev = 'data/structured/devices-not-present/'
        fpath_dev = 'data/structured/devices-present/'
        probabilities_file = 'data/structured/probabilities/probabilities.csv'

        # Execute the function under test
        result = vm.calculate_velocity_stressors(fpath_nodev, fpath_dev, probabilities_file)

        # Assertions to verify the expected outcomes
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 6)


# class TestRunVelocityStressor(unittest.TestCase):
#     @patch('seat.velocity_module.calculate_velocity_stressors')
#     @patch('seat.stressor_utils.secondary_constraint_geotiff_to_numpy')
#     @patch('seat.stressor_utils.create_raster')
#     @patch('seat.stressor_utils.numpy_array_to_raster')
#     @patch('os.path.exists')
#     @patch('os.makedirs')
#     def test_run_velocity_stressor_with_basic_setup(self, mock_makedirs, mock_path_exists, mock_array_to_raster, mock_create_raster, mock_secondary_constraint, mock_calculate_velocity_stressors):
#         # Example shape and data for mock numpy arrays
#         example_shape = (5, 5)
#         example_data = np.random.rand(*example_shape)  # Generate random data

#         # Mock numpy arrays with example data and shape
#         placeholder_numpy_arrays = [MagicMock(spec=np.ndarray, shape=example_shape) for _ in range(8)]
#         for mock_array in placeholder_numpy_arrays:
#             mock_array.mean.return_value = example_data.mean()
#             mock_array.max.return_value = example_data.max()
#             type(mock_array).shape = PropertyMock(return_value=example_shape)

#         # Correct return values for rx, ry to be numpy arrays
#         rx = np.arange(5)
#         ry = np.arange(5)

#         mock_calculate_velocity_stressors.return_value = (placeholder_numpy_arrays, rx, ry, 1, 1, 'structured')

#         # Define inputs
#         dev_present_file = 'data/structured/devices-present/'
#         dev_notpresent_file = 'data/structured/devices-not-present/'
#         probabilities_file = 'data/structured/probabilities/probabilities.csv'
#         crs = 4326
#         output_path = 'data/output'
#         receptor_filename = None
#         secondary_constraint_filename = None

#         # Call the function under test
#         result = vm.run_velocity_stressor(dev_present_file, dev_notpresent_file, probabilities_file, crs, output_path, receptor_filename, secondary_constraint_filename)


#         # Assert that necessary directories were checked/created
#         mock_makedirs.assert_called_with(output_path, exist_ok=True)
        
#         # Additional assertions
#         self.assertIsInstance(result, dict)

if __name__ == '__main__':
    unittest.main()
