import sys
import os
import netCDF4
import unittest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from qgis.core import QgsApplication
import configparser


# Get the directory in which the current script is located
script_dir = os.path.dirname(os.path.realpath(__file__))

# Import seat
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# fmt: off
from seat import stressor_receptor_calc as sr
# fmt: on


class TestStressorReceptorCalcModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Class method to set up test environment.
        """
        # Create a mock CSV file for testing
        cls.mock_csv_file = 'mock_style_file.csv'
        cls.create_mock_csv(cls.mock_csv_file)

        cls.mock_iface = MagicMock()  # Mock the iface
        cls.stressor_receptor_calc = sr.StressorReceptorCalc(cls.mock_iface)
        cls.stressor_receptor_calc.dlg = MagicMock()

    @classmethod
    def tearDownClass(cls):
        """
        Class method to clean up after tests.
        """
        # Other teardown code...

        # Remove mock CSV file
        if os.path.exists(cls.mock_csv_file):
            os.remove(cls.mock_csv_file)

    @staticmethod
    def create_mock_csv(filename):
        """
        Create a mock CSV file with predefined content.
        """
        data = {
            'Type': ['Type1', 'Type2', 'Type3'],
            'Value': [10, 20, 30]
        }
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

    @patch('seat.stressor_receptor_calc.QFileDialog.getExistingDirectory')
    def test_select_folder(self, mock_getExistingDirectory):
        """
        Test selecting a folder using select_file with file=False.
        """
        # Mock folder names for testing
        mock_folder_name_present = "C:/DeviceFolderPresent"
        mock_folder_name_not_present = "C:/DeviceFolderNotPresent"

        # Test for 'not present' condition
        mock_getExistingDirectory.return_value = mock_folder_name_not_present
        folder_name = self.stressor_receptor_calc.select_file(file=False)
        self.assertEqual(folder_name, mock_folder_name_not_present)

        # Reset the mock to test for 'present' condition
        mock_getExistingDirectory.reset_mock()

        # Test for 'present' condition
        mock_getExistingDirectory.return_value = mock_folder_name_present
        folder_name = self.stressor_receptor_calc.select_file(file=False)
        self.assertEqual(folder_name, mock_folder_name_present)

    def test_read_style_files(self):
        """
        Test the read_style_files method.
        """
        stressor_receptor_calc = sr.StressorReceptorCalc(None)
        result_df = stressor_receptor_calc.read_style_files(self.mock_csv_file)

        # Adjusting expected_df to have the same index name as result_df
        expected_df = pd.DataFrame({'Value': [10, 20, 30]}, index=pd.Index(['Type1', 'Type2', 'Type3'], name='Type'))

        pd.testing.assert_frame_equal(result_df, expected_df)

    @patch('seat.stressor_receptor_calc.QFileDialog.getOpenFileName')
    def test_select_file(self, mock_getOpenFileName):
        """
        Test the select_file method.
        """
        # Mock file name for testing
        mock_filename = "C:/TestFile.csv"

        # Set up the mock return value
        mock_getOpenFileName.return_value = (mock_filename, '')

        # Call the select_file method and capture the return value
        filename = self.stressor_receptor_calc.select_file(file_filter="*.csv")

        # Check that the return value matches the mock filename
        self.assertEqual(filename, mock_filename)

    def test_copy_shear_input_to_velocity(self):
        """Test the copy_shear_input_to_velocity method."""
        # Create mock dialog attributes
        self.stressor_receptor_calc.dlg = MagicMock()
        self.stressor_receptor_calc.dlg.shear_device_present = MagicMock()
        self.stressor_receptor_calc.dlg.shear_device_not_present = MagicMock()
        self.stressor_receptor_calc.dlg.shear_probabilities_file = MagicMock()
        self.stressor_receptor_calc.dlg.shear_risk_file = MagicMock()
        self.stressor_receptor_calc.dlg.shear_aoi_file = MagicMock()
        self.stressor_receptor_calc.dlg.velocity_device_present = MagicMock()
        self.stressor_receptor_calc.dlg.velocity_device_not_present = MagicMock()
        self.stressor_receptor_calc.dlg.velocity_probabilities_file = MagicMock()
        self.stressor_receptor_calc.dlg.velocity_risk_file = MagicMock()
        self.stressor_receptor_calc.dlg.velocity_aoi_file = MagicMock()

        # Set return values for shear fields
        self.stressor_receptor_calc.dlg.shear_device_present.text.return_value = "shear_device_present_path"
        self.stressor_receptor_calc.dlg.shear_device_not_present.text.return_value = "shear_device_not_present_path"
        self.stressor_receptor_calc.dlg.shear_probabilities_file.text.return_value = "shear_probabilities_file"
        self.stressor_receptor_calc.dlg.shear_risk_file.text.return_value = "shear_risk_file"
        self.stressor_receptor_calc.dlg.shear_aoi_file.text.return_value = "shear_aoi_file"

        self.stressor_receptor_calc.copy_shear_input_to_velocity()

        # Verify the velocity fields were set correctly
        self.stressor_receptor_calc.dlg.velocity_device_present.setText.assert_called_once_with("shear_device_present_path")
        self.stressor_receptor_calc.dlg.velocity_device_not_present.setText.assert_called_once_with("shear_device_not_present_path")
        self.stressor_receptor_calc.dlg.velocity_probabilities_file.setText.assert_called_once_with("shear_probabilities_file")
        self.stressor_receptor_calc.dlg.velocity_aoi_file.setText.assert_called_once_with("shear_aoi_file")

    @patch('seat.stressor_receptor_calc.QgsProjectionSelectionDialog')
    @patch('seat.stressor_receptor_calc.QgsCoordinateReferenceSystem')
    def test_select_crs(self, mock_QgsCoordinateReferenceSystem, mock_QgsProjectionSelectionDialog):
        """
        Test the select_crs method.
        """
        # Mocking QgsCoordinateReferenceSystem and QgsProjectionSelectionDialog
        mock_crs = MagicMock()
        mock_crs.authid.return_value = "EPSG:4326"

        mock_proj_selector = MagicMock()
        mock_proj_selector.exec.return_value = True
        mock_proj_selector.crs.return_value = mock_crs

        mock_QgsProjectionSelectionDialog.return_value = mock_proj_selector
        self.stressor_receptor_calc.dlg.crs.reset_mock()
        # Execute the function
        self.stressor_receptor_calc.select_crs()

        # Verify if the setText method of self.stressor_receptor_calc.dlg.crs was called with the expected CRS ID
        self.stressor_receptor_calc.dlg.crs.setText.assert_called_once_with("4326")

    @patch('os.path.exists')
    def test_test_exists(self, mock_exists):
        """
        Test the test_exists method.
        """
        line_edit = MagicMock()

        # Case: file exists
        mock_exists.return_value = True
        self.stressor_receptor_calc.test_exists(line_edit, "existing_path", "File")
        line_edit.setText.assert_called_once_with("existing_path")
        line_edit.setStyleSheet.assert_called_once_with("color: black;")

        # Reset mocks
        line_edit.reset_mock()

        # Case: file does not exist
        mock_exists.return_value = False
        self.stressor_receptor_calc.test_exists(line_edit, "non_existing_path", "File")
        line_edit.setText.assert_called_once_with("File not Found")
        line_edit.setStyleSheet.assert_called_once_with("color: red;")

        # Reset mocks
        line_edit.reset_mock()

        # Case: file path is empty
        self.stressor_receptor_calc.test_exists(line_edit, "", "File")
        line_edit.setStyleSheet.assert_called_once_with("color: black;")


    @patch('seat.stressor_receptor_calc.QFileDialog.getOpenFileName')
    @patch('seat.stressor_receptor_calc.configparser.ConfigParser')
    @patch.object(sr.StressorReceptorCalc, 'test_exists')
    def test_select_and_load_in(self, mock_test_exists, mock_configparser, mock_getOpenFileName):
        """Test the select_and_load_in method."""
        # Mock file name and configparser
        mock_filename = "config.ini"
        mock_getOpenFileName.return_value = (mock_filename, None)

        mock_config = MagicMock()
        mock_config.get.side_effect = lambda section, key: f"{section} {key}"
        mock_configparser.return_value = mock_config

        # Create a mock instance of the dialog
        self.stressor_receptor_calc.dlg = MagicMock()
        self.stressor_receptor_calc.dlg.shear_device_present = MagicMock()
        self.stressor_receptor_calc.dlg.shear_device_not_present = MagicMock()
        self.stressor_receptor_calc.dlg.shear_probabilities_file = MagicMock()
        self.stressor_receptor_calc.dlg.shear_grain_size_file = MagicMock()
        self.stressor_receptor_calc.dlg.shear_aoi_file = MagicMock()
        self.stressor_receptor_calc.dlg.shear_averaging_combobox = MagicMock()
        self.stressor_receptor_calc.dlg.velocity_device_present = MagicMock()
        self.stressor_receptor_calc.dlg.velocity_device_not_present = MagicMock()
        self.stressor_receptor_calc.dlg.velocity_probabilities_file = MagicMock()
        self.stressor_receptor_calc.dlg.velocity_threshold_file = MagicMock()
        self.stressor_receptor_calc.dlg.velocity_aoi_file = MagicMock()
        self.stressor_receptor_calc.dlg.velocity_averaging_combobox = MagicMock()
        self.stressor_receptor_calc.dlg.paracousti_device_present = MagicMock()
        self.stressor_receptor_calc.dlg.paracousti_device_not_present = MagicMock()
        self.stressor_receptor_calc.dlg.paracousti_probabilities_file = MagicMock()
        self.stressor_receptor_calc.dlg.paracousti_aoi_file = MagicMock()
        self.stressor_receptor_calc.dlg.paracousti_species_directory = MagicMock()
        self.stressor_receptor_calc.dlg.paracousti_averaging_combobox = MagicMock()
        self.stressor_receptor_calc.dlg.paracousti_threshold_value = MagicMock()
        self.stressor_receptor_calc.dlg.power_files = MagicMock()
        self.stressor_receptor_calc.dlg.power_probabilities_file = MagicMock()
        self.stressor_receptor_calc.dlg.output_stylefile = MagicMock()
        self.stressor_receptor_calc.dlg.crs = MagicMock()
        self.stressor_receptor_calc.dlg.output_folder = MagicMock()

        # Execute the function
        self.stressor_receptor_calc.select_and_load_in()

        calls = [
            ('shear_device_present', 'Input shear stress device present filepath', 'Directory'),
            ('shear_device_not_present', 'Input shear stress device not present filepath', 'Directory'),
            ('shear_probabilities_file', 'Input shear stress probabilities file', 'File'),
            ('shear_grain_size_file', 'Input shear stress grain size file', 'File'),
            ('shear_aoi_file', 'Input shear stress area of interest file', 'File'),
            ('velocity_device_present', 'Input velocity device present filepath', 'Directory'),
            ('velocity_device_not_present', 'Input velocity device not present filepath', 'Directory'),
            ('velocity_probabilities_file', 'Input velocity probabilities file', 'File'),
            ('velocity_threshold_file', 'Input velocity threshold file', 'File'),
            ('velocity_aoi_file', 'Input velocity area of interest file', 'File'),
            ('paracousti_device_present', 'Input paracousti device present filepath', 'Directory'),
            ('paracousti_device_not_present', 'Input paracousti device not present filepath', 'Directory'),
            ('paracousti_probabilities_file', 'Input paracousti probabilities file', 'File'),
            ('paracousti_aoi_file', 'Input paracousti area of interest file', 'File'),
            ('paracousti_species_directory', 'Input paracousti species filepath', 'Directory'),
            ('power_files', 'Input power files filepath', 'Directory'),
            ('power_probabilities_file', 'Input power probabilities file', 'File'),
            ('output_stylefile', 'Input output style files', 'File')
        ]

        for dlg_attr, config_value, fin_type in calls:
            attr = getattr(self.stressor_receptor_calc.dlg, dlg_attr)
            mock_test_exists.assert_any_call(attr, config_value, fin_type)

        # Check that non-file values are set correctly
        self.stressor_receptor_calc.dlg.shear_averaging_combobox.setCurrentText.assert_called_once_with("Input shear stress averaging")
        self.stressor_receptor_calc.dlg.velocity_averaging_combobox.setCurrentText.assert_called_once_with("Input velocity Averaging")
        self.stressor_receptor_calc.dlg.paracousti_averaging_combobox.setCurrentText.assert_called_once_with("Input paracousti averaging")
        self.stressor_receptor_calc.dlg.paracousti_threshold_value.setText.assert_called_once_with("Input paracousti_threshold_value")
        self.stressor_receptor_calc.dlg.crs.setText.assert_called_once_with("Input coordinate reference system")
        self.stressor_receptor_calc.dlg.output_folder.setText.assert_called_once_with("Output output filepath")


    @patch('seat.stressor_receptor_calc.QFileDialog.getSaveFileName')
    def test_save_in(self, mock_getSaveFileName):
        """
        Test the save_in method.
        """
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        mock_getSaveFileName.return_value = (temp_file.name, None)

        # Mock text methods
        self.stressor_receptor_calc.dlg.shear_device_present.text.return_value = "shear_device_present_path"
        self.stressor_receptor_calc.dlg.shear_device_not_present.text.return_value = "shear_device_not_present_path"
        self.stressor_receptor_calc.dlg.shear_averaging_combobox.currentText.return_value = "shear_averaging"
        self.stressor_receptor_calc.dlg.shear_probabilities_file.text.return_value = "shear_probabilities_file"
        self.stressor_receptor_calc.dlg.shear_grain_size_file.text.return_value = "shear_grain_size_file"
        self.stressor_receptor_calc.dlg.shear_aoi_file.text.return_value = "shear_aoi_file"
        self.stressor_receptor_calc.dlg.velocity_device_present.text.return_value = "velocity_device_present_path"
        self.stressor_receptor_calc.dlg.velocity_device_not_present.text.return_value = "velocity_device_not_present_path"
        self.stressor_receptor_calc.dlg.velocity_averaging_combobox.currentText.return_value = "velocity_averaging"
        self.stressor_receptor_calc.dlg.velocity_probabilities_file.text.return_value = "velocity_probabilities_file"
        self.stressor_receptor_calc.dlg.velocity_threshold_file.text.return_value = "velocity_threshold_file"
        self.stressor_receptor_calc.dlg.velocity_aoi_file.text.return_value = "velocity_aoi_file"
        self.stressor_receptor_calc.dlg.paracousti_device_present.text.return_value = "paracousti_device_present_path"
        self.stressor_receptor_calc.dlg.paracousti_device_not_present.text.return_value = "paracousti_device_not_present_path"
        self.stressor_receptor_calc.dlg.paracousti_averaging_combobox.currentText.return_value = "paracousti_averaging"
        self.stressor_receptor_calc.dlg.paracousti_probabilities_file.text.return_value = "paracousti_probabilities_file"
        self.stressor_receptor_calc.dlg.paracousti_aoi_file.text.return_value = "paracousti_aoi_file"
        self.stressor_receptor_calc.dlg.paracousti_species_directory.text.return_value = "paracousti_species_directory"
        self.stressor_receptor_calc.dlg.paracousti_metric_selection_combobox.currentText.return_value = "paracousti_metric"
        self.stressor_receptor_calc.dlg.paracousti_weighting_combobox.currentText.return_value = "paracousti_weighting"
        self.stressor_receptor_calc.dlg.paracousti_species_grid_resolution.text.return_value = "paracousti_species_grid_resolution"
        self.stressor_receptor_calc.dlg.power_files.text.return_value = "power_files_path"
        self.stressor_receptor_calc.dlg.power_probabilities_file.text.return_value = "power_probabilities_file"
        self.stressor_receptor_calc.dlg.crs.text.return_value = "coordinate_system"
        self.stressor_receptor_calc.dlg.output_stylefile.text.return_value = "output_style_files"
        self.stressor_receptor_calc.dlg.output_folder.text.return_value = "output_folder_path"

        # Execute the function
        self.stressor_receptor_calc.save_in()

        # Read the temporary file and verify contents
        config = configparser.ConfigParser()
        config.read(temp_file.name)

        self.assertEqual(config["Input"]["shear stress device present filepath"], "shear_device_present_path")
        self.assertEqual(config["Input"]["shear stress device not present filepath"], "shear_device_not_present_path")
        self.assertEqual(config["Input"]["shear stress averaging"], "shear_averaging")
        self.assertEqual(config["Input"]["shear stress probabilities file"], "shear_probabilities_file")
        self.assertEqual(config["Input"]["shear stress grain size file"], "shear_grain_size_file")
        self.assertEqual(config["Input"]["shear stress area of interest file"], "shear_aoi_file")
        self.assertEqual(config["Input"]["velocity device present filepath"], "velocity_device_present_path")
        self.assertEqual(config["Input"]["velocity device not present filepath"], "velocity_device_not_present_path")
        self.assertEqual(config["Input"]["velocity averaging"], "velocity_averaging")
        self.assertEqual(config["Input"]["velocity probabilities file"], "velocity_probabilities_file")
        self.assertEqual(config["Input"]["velocity threshold file"], "velocity_threshold_file")
        self.assertEqual(config["Input"]["velocity area of interest file"], "velocity_aoi_file")
        self.assertEqual(config["Input"]["paracousti device present filepath"], "paracousti_device_present_path")
        self.assertEqual(config["Input"]["paracousti device not present filepath"], "paracousti_device_not_present_path")
        self.assertEqual(config["Input"]["paracousti averaging"], "paracousti_averaging")
        self.assertEqual(config["Input"]["paracousti probabilities file"], "paracousti_probabilities_file")
        self.assertEqual(config["Input"]["paracousti area of interest file"], "paracousti_aoi_file")
        self.assertEqual(config["Input"]["paracousti species filepath"], "paracousti_species_directory")
        self.assertEqual(config["Input"]["paracousti metric"], "paracousti_metric")
        self.assertEqual(config["Input"]["paracousti weighting"], "paracousti_weighting")
        self.assertEqual(config["Input"]["paracousti_species_grid_resolution"], "paracousti_species_grid_resolution")
        self.assertEqual(config["Input"]["power files filepath"], "power_files_path")
        self.assertEqual(config["Input"]["power probabilities file"], "power_probabilities_file")
        self.assertEqual(config["Input"]["coordinate reference system"], "coordinate_system")
        self.assertEqual(config["Input"]["output style files"], "output_style_files")
        self.assertEqual(config["Output"]["output filepath"], "output_folder_path")

        # Cleanup
        temp_file.close()
        os.remove(temp_file.name)

    @patch('seat.stressor_receptor_calc.QgsProject.instance')
    @patch('seat.stressor_receptor_calc.QgsRasterLayer')
    def test_add_layer(self, mock_QgsRasterLayer, mock_QgsProject_instance):
        # Set up mock objects
        mock_layer = MagicMock()
        mock_QgsRasterLayer.return_value = mock_layer
        
        mock_project = MagicMock()
        mock_tree_root = MagicMock()
        mock_project.layerTreeRoot.return_value = mock_tree_root
        mock_QgsProject_instance.return_value = mock_project

        # Mock parameters
        fpath = "test_fpath.tif"
        stylepath = None
        group = MagicMock()

        # Call the function with root=None to test the default behavior
        self.stressor_receptor_calc.add_layer(fpath, stylepath=stylepath, group=group)

        # Assertions
        mock_QgsRasterLayer.assert_called_once_with(fpath, 'test_fpath')
        mock_project.addMapLayer.assert_called_once_with(mock_layer)
        mock_tree_root.findLayer.assert_called_once_with(mock_layer.id())
        mock_tree_root.removeChildNode.assert_called_once()
        group.insertChildNode.assert_called_once()

    @patch('seat.stressor_receptor_calc.QgsRasterLayer')
    @patch('seat.stressor_receptor_calc.QgsProject.instance')
    def test_add_layer(self, mock_QgsProject_instance, mock_QgsRasterLayer):
        # Mocking QgsRasterLayer and QgsProject.instance
        mock_layer = MagicMock()
        mock_QgsRasterLayer.return_value = mock_layer
        mock_project = MagicMock()
        mock_QgsProject_instance.return_value = mock_project
        mock_project.addMapLayer.return_value = mock_layer

        # Set return values and mock methods
        mock_layer.loadNamedStyle = MagicMock()
        mock_layer.triggerRepaint = MagicMock()
        mock_layer.reload = MagicMock()

        # Call the function
        fpath = "test_fpath.tif"
        stylepath = "test_style.qml"

        root = MagicMock()
        group = MagicMock()

        self.stressor_receptor_calc.add_layer(fpath, stylepath, root=root, group=group)

        # Assertions
        mock_layer.loadNamedStyle.assert_called_once_with(stylepath)
        mock_project.addMapLayer.assert_called_once_with(mock_layer)

        # Test the function without stylepath
        mock_layer.reset_mock()
        mock_project.reset_mock()

        self.stressor_receptor_calc.add_layer(fpath, None, root=root, group=group)

        # Assert that the style was not applied
        mock_layer.loadNamedStyle.assert_not_called()

    @patch.object(sr.StressorReceptorCalc, 'select_file')
    def test_select_folder_module(self, mock_select_file):
        """Test the select_folder_module method."""
        # Set up the mock return value for select_file
        mock_directory = "C:/path/to/directory"
        mock_select_file.return_value = mock_directory

        # Create a fresh mock dialog for each test
        self.stressor_receptor_calc.dlg = MagicMock()

        # Test cases for each module and option
        test_cases = [
            ('shear', 'device_present', 'shear_device_present'),
            # ('shear', 'device_not_present', 'shear_device_not_present'),
            # ('velocity', 'device_present', 'velocity_device_present'),
            # ('velocity', 'device_not_present', 'velocity_device_not_present'),
            # ('paracousti', 'device_present', 'paracousti_device_present'),
            # ('paracousti', 'device_not_present', 'paracousti_device_not_present'),
            # ('paracousti', 'species_directory', 'paracousti_species_directory'),
            # ('power', None, 'power_files'),
            # ('output', None, 'output_folder'),
        ]

        for module, option, attr_name in test_cases:
            with self.subTest(module=module, option=option):
                # Create a new mock for each line edit
                line_edit = MagicMock()
                setattr(self.stressor_receptor_calc.dlg, attr_name, line_edit)

                # Call the function with the test case parameters
                self.stressor_receptor_calc.select_folder_module(module=module, option=option)

                # Assert that setText was called with the mock directory path
                line_edit.setText.assert_called_once_with(mock_directory)
                line_edit.setStyleSheet.assert_called_once_with("color: black;")

    @patch.object(sr.StressorReceptorCalc, 'select_file')
    def test_select_files_module(self, mock_select_file):
        """Test the select_files_module method."""
        # Create a fresh mock dialog for each test to avoid interference
        self.stressor_receptor_calc.dlg = MagicMock()

        # Test cases for each module and option with their expected file filters
        test_cases = [
            ('shear', 'probabilities_file', 'shear_probabilities_file', '*.csv'),
            ('shear', 'grain_size_file', 'shear_grain_size_file', '*.tif; *.csv'),
            ('shear', 'risk_file', 'shear_aoi_file', '*.tif'),
            ('velocity', 'probabilities_file', 'velocity_probabilities_file', '*.csv'),
            ('velocity', 'thresholds', 'velocity_threshold_file', '*.tif; *.csv'),
            ('velocity', 'risk_file', 'velocity_aoi_file', '*.tif'),
            ('paracousti', 'probabilities_file', 'paracousti_probabilities_file', '*.csv'),
            ('paracousti', 'risk_file', 'paracousti_aoi_file', '*.tif'),
            ('power', None, 'power_probabilities_file', '*.csv'),
            ('style_files', None, 'output_stylefile', '*.csv'),
        ]

        # Mock file path that will be returned
        mock_file = "C:/path/to/file.csv"
        mock_select_file.return_value = mock_file

        for module, option, attr_name, file_filter in test_cases:
            with self.subTest(module=module, option=option):
                # Reset the mock for each test case
                mock_select_file.reset_mock()
                
                # Create a new mock for each line edit
                line_edit = MagicMock()
                setattr(self.stressor_receptor_calc.dlg, attr_name, line_edit)

                # Call the function with the test case parameters
                self.stressor_receptor_calc.select_files_module(module=module, option=option)

                # Verify select_file was called with the correct file filter
                mock_select_file.assert_called_once_with(file_filter=file_filter)
                
                # Assert that setText was called with the mock file path
                line_edit.setText.assert_called_once_with(mock_file)
                line_edit.setStyleSheet.assert_called_once_with("color: black;")


def run_all():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestStressorReceptorCalcModule))
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == '__main__':
    run_all()