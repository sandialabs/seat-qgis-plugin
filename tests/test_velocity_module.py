import sys
import os
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, PropertyMock
import netCDF4
from os.path import join
import shutil

# Get the directory in which the current script is located
script_dir = os.path.dirname(os.path.realpath(__file__))

# Import seat
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# fmt: off
from seat.modules import velocity_module as vm

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
        mock_listdir.side_effect = lambda x: ['last_2_runs.nc'] if 'devices-present' in x else ['last_2_runs.nc']

        # Setup MagicMock for the netCDF dataset
        mock_u_data = np.random.rand(5, 5)  # Random data for demonstration
        mock_v_data = np.random.rand(5, 5)
        mock_x_data = np.arange(5)
        mock_y_data = np.arange(5)

        mock_uvar = MagicMock()
        mock_uvar.__getitem__.return_value = mock_u_data
        mock_vvar = MagicMock()
        mock_vvar.__getitem__.return_value = mock_v_data
        mock_xvar = MagicMock()
        mock_xvar.__getitem__.return_value = mock_x_data
        mock_yvar = MagicMock()
        mock_yvar.__getitem__.return_value = mock_y_data

        mock_ds_instance = MagicMock()
        mock_ds_instance.variables = {
            'U1': mock_uvar,
            'V1': mock_vvar,
            'X1': mock_xvar,
            'Y1': mock_yvar
        }
        mock_dataset.return_value = mock_ds_instance

        # Define inputs using os.path.join
        current_working_directory = os.getcwd()
        fpath_nodev = os.path.join(current_working_directory, 'tests', 'data', 'structured', 'devices-not-present')
        fpath_dev = os.path.join(current_working_directory, 'tests', 'data', 'structured', 'devices-present')
        probabilities_file = os.path.join(current_working_directory, 'tests', 'data', 'structured', 'probabilities', 'probabilities.csv')

        # Execute the function under test
        result = vm.calculate_velocity_stressors(fpath_nodev, fpath_dev, probabilities_file)

        # Assertions to verify the expected outcomes
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 6)

class TestRunVelocityStressor(unittest.TestCase):

    def test_run_velocity_stressor_with_basic_setup(self, ):
        
        # Define inputs
        dev_present_file = join(script_dir, "data","structured","devices-present")
        dev_not_present_file = join(script_dir, "data","structured","devices-not-present")
        probabilities_file = join(script_dir, "data","structured","probabilities","probabilities.csv")
        receptor_file = join(script_dir, "data","structured","receptor","grain_size_receptor.csv")
        crs = 4326
        secondary_constraint_filename = None
        output_path = join(script_dir, "data", "output")
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Call the function under test
        result = vm.run_velocity_stressor(dev_present_file, dev_not_present_file, probabilities_file, crs, output_path, receptor_file, secondary_constraint_filename)

        # Additional assertions
        self.assertIsInstance(result, dict)

        # Now remove the directory
        shutil.rmtree(output_path)

if __name__ == '__main__':
    unittest.main()
