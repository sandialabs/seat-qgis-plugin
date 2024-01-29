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

    @patch('seat.stressor_receptor_calc.QFileDialog.getOpenFileName')
    def test_select_style_files(self, mock_getOpenFileName):
        """
        Test the select_style_files method.
        """
        # Mock file name for testing
        mock_filename = self.mock_csv_file
        mock_getOpenFileName.return_value = (mock_filename, None)

        self.stressor_receptor_calc.dlg.output_stylefile.reset_mock()
        self.stressor_receptor_calc.select_style_files()
        # Check that the setText method of self.stressor_receptor_calc.dlg.output_stylefile was called exactly once with mock_filename as its argument
        self.stressor_receptor_calc.dlg.output_stylefile.setText.assert_called_once_with(mock_filename)


    @patch('seat.stressor_receptor_calc.QFileDialog.getExistingDirectory')
    def test_select_device_folder(self, mock_getExistingDirectory):
        """
        Test the select_device_folder method.
        """
        # Mock folder names for testing
        mock_folder_name_present = "C:/DeviceFolderPresent"
        mock_folder_name_not_present = "C:/DeviceFolderNotPresent"

        # Test for 'not present' condition
        mock_getExistingDirectory.return_value = mock_folder_name_not_present
        self.stressor_receptor_calc.dlg.device_not_present.reset_mock()
        self.stressor_receptor_calc.select_device_folder("not present")
        self.stressor_receptor_calc.dlg.device_not_present.setText.assert_called_once_with(mock_folder_name_not_present)

        # Reset the mock to test for 'present' condition
        mock_getExistingDirectory.reset_mock()
        self.stressor_receptor_calc.dlg.device_not_present.setText.reset_mock()

        # Test for 'present' condition
        mock_getExistingDirectory.return_value = mock_folder_name_present
        self.stressor_receptor_calc.dlg.device_present.reset_mock()
        self.stressor_receptor_calc.select_device_folder("present")
        self.stressor_receptor_calc.dlg.device_present.setText.assert_called_once_with(mock_folder_name_present)

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
    def test_select_probabilities_file(self, mock_getOpenFileName):
        """
        Test the select_probabilities_file method.
        """
        # mock object created to simulate the behavior of QFileDialog.getOpenFileName.
        mock_filename="C:/Users/sterl/Codes/seat-qgis-plugin/tests/data/structured/probabilities/probabilities.csv"
        mock_getOpenFileName.return_value = (mock_filename, None)
        self.stressor_receptor_calc.dlg.probabilities_file.reset_mock()

        self.stressor_receptor_calc.select_probabilities_file()
        # checks that the setText method of self.stressor_receptor_calc.dlg.  probabilities_file was called exactly once with mock_filename as its argument. 
        self.stressor_receptor_calc.dlg.probabilities_file.setText.assert_called_once_with(mock_filename)

    @patch('seat.stressor_receptor_calc.QFileDialog.getExistingDirectory')
    def test_select_power_files_folder(self, mock_getExistingDirectory):
        """
        Test the select_power_files_folder method.
        """
        # Mock folder names for testing
        mock_folder_power_folder = "C:/DevicePowerFolder"
        mock_getExistingDirectory.return_value = mock_folder_power_folder
        self.stressor_receptor_calc.dlg.power_files.reset_mock()

        # Test for 'not present' condition
        self.stressor_receptor_calc.select_power_files_folder()
        self.stressor_receptor_calc.dlg.power_files.setText.assert_called_once_with(mock_folder_power_folder)

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

    @patch('seat.stressor_receptor_calc.QFileDialog.getOpenFileName')
    def test_select_receptor_file(self, mock_getOpenFileName):
        """
        Test the select_receptor_file method.
        """
        # Mock file name for testing
        mock_filename = "receptor.tif"
        mock_getOpenFileName.return_value = (mock_filename, None)
        self.stressor_receptor_calc.dlg.receptor_file.reset_mock()

        self.stressor_receptor_calc.select_receptor_file()
        # Check that the setText method of self.stressor_receptor_calc.dlg.receptor_file was called exactly once with mock_filename as its argument
        self.stressor_receptor_calc.dlg.receptor_file.setText.assert_called_once_with(mock_filename)

    @patch('seat.stressor_receptor_calc.QFileDialog.getExistingDirectory')
    def test_select_secondary_constraint_folder(self, mock_getExistingDirectory):
        """
        Test the select_secondary_constraint_folder method.
        """
        # Mock folder name for testing
        mock_folder_name = "secondaryConstraints"
        mock_getExistingDirectory.return_value = mock_folder_name
        self.stressor_receptor_calc.dlg.sc_file.reset_mock()

        self.stressor_receptor_calc.select_secondary_constraint_folder()
        # Check that the setText method of self.stressor_receptor_calc.dlg.sc_file was called exactly once with mock_folder_name as its argument
        self.stressor_receptor_calc.dlg.sc_file.setText.assert_called_once_with(mock_folder_name)

    @patch('seat.stressor_receptor_calc.QFileDialog.getExistingDirectory')
    def test_select_output_folder(self, mock_getExistingDirectory):
        """
        Test the select_output_folder method.
        """
        # Mock folder name for testing
        mock_folder_name = "output"
        mock_getExistingDirectory.return_value = mock_folder_name
        self.stressor_receptor_calc.dlg.output_folder.reset_mock()

        self.stressor_receptor_calc.select_output_folder()
        # Check that the setText method of self.stressor_receptor_calc.dlg.output_folder was called exactly once with mock_folder_name as its argument
        self.stressor_receptor_calc.dlg.output_folder.setText.assert_called_once_with(mock_folder_name)


    @patch('seat.stressor_receptor_calc.QFileDialog.getOpenFileName')
    @patch('seat.stressor_receptor_calc.configparser.ConfigParser')
    def test_select_and_load_in(self, mock_configparser, mock_getOpenFileName):
        """
        Test the select_and_load_in method.
        """
        # Mock file name and configparser
        mock_filename = "config.ini"
        mock_getOpenFileName.return_value = (mock_filename, None)

        mock_config = MagicMock()
        mock_config.get.side_effect = lambda section, key: f"{section} {key}"
        mock_configparser.return_value = mock_config

        # Reset calls for each mocked method
        self.stressor_receptor_calc.dlg.device_present.reset_mock()
        self.stressor_receptor_calc.dlg.device_not_present.reset_mock()
        self.stressor_receptor_calc.dlg.probabilities_file.reset_mock()
        self.stressor_receptor_calc.dlg.power_files.reset_mock()
        self.stressor_receptor_calc.dlg.receptor_file.reset_mock()
        self.stressor_receptor_calc.dlg.sc_file.reset_mock()
        self.stressor_receptor_calc.dlg.stressor_comboBox.setCurrentText.reset_mock()
        self.stressor_receptor_calc.dlg.crs.reset_mock()
        self.stressor_receptor_calc.dlg.output_folder.reset_mock()
        self.stressor_receptor_calc.dlg.output_stylefile.reset_mock()

        # Execute the function
        self.stressor_receptor_calc.select_and_load_in()

        # Verify if the text fields are set correctly
        self.stressor_receptor_calc.dlg.device_present.setText.assert_called_once_with("Input device present filepath")
        self.stressor_receptor_calc.dlg.device_not_present.setText.assert_called_once_with("Input device not present filepath")
        self.stressor_receptor_calc.dlg.probabilities_file.setText.assert_called_once_with("Input probabilities filepath")
        self.stressor_receptor_calc.dlg.power_files.setText.assert_called_once_with("Input power files filepath")
        self.stressor_receptor_calc.dlg.receptor_file.setText.assert_called_once_with("Input receptor filepath")
        self.stressor_receptor_calc.dlg.sc_file.setText.assert_called_once_with("Input secondary constraint filepath")
        self.stressor_receptor_calc.dlg.stressor_comboBox.setCurrentText.assert_called_once_with("Input stressor variable")
        self.stressor_receptor_calc.dlg.crs.setText.assert_called_once_with("Input coordinate reference system")
        self.stressor_receptor_calc.dlg.output_folder.setText.assert_called_once_with("Output output filepath")
        self.stressor_receptor_calc.dlg.output_stylefile.setText.assert_called_once_with("Input output style files")


    @patch('seat.stressor_receptor_calc.QFileDialog.getSaveFileName')
    def test_save_in(self, mock_getSaveFileName):
        """
        Test the save_in method.
        """
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        mock_getSaveFileName.return_value = (temp_file.name, None)

        # Mock text methods
        self.stressor_receptor_calc.dlg.device_present.text.return_value = "device_present_path"
        self.stressor_receptor_calc.dlg.device_not_present.text.return_value = "device_not_present_path"
        self.stressor_receptor_calc.dlg.probabilities_file.text.return_value = "probabilities_path"
        self.stressor_receptor_calc.dlg.power_files.text.return_value = "power_files_path"
        self.stressor_receptor_calc.dlg.receptor_file.text.return_value = "receptor_path"
        self.stressor_receptor_calc.dlg.sc_file.text.return_value = "secondary_constraint_path"
        self.stressor_receptor_calc.dlg.stressor_comboBox.currentText.return_value = "stressor_variable"
        self.stressor_receptor_calc.dlg.crs.text.return_value = "coordinate_system"
        self.stressor_receptor_calc.dlg.output_folder.text.return_value = "output_folder_path"
        self.stressor_receptor_calc.dlg.output_stylefile.text.return_value = "output_style_files"

        # Execute the function
        self.stressor_receptor_calc.save_in()

        # Read the temporary file and verify contents
        config = configparser.ConfigParser()
        config.read(temp_file.name)

        self.assertEqual(config["Input"]["device present filepath"], "device_present_path")
        self.assertEqual(config["Input"]["device not present filepath"], "device_not_present_path")
        self.assertEqual(config["Input"]["probabilities filepath"], "probabilities_path")
        self.assertEqual(config["Input"]["power files filepath"], "power_files_path")
        self.assertEqual(config["Input"]["receptor filepath"], "receptor_path")
        self.assertEqual(config["Input"]["secondary constraint filepath"], "secondary_constraint_path")
        self.assertEqual(config["Input"]["stressor variable"], "stressor_variable")
        self.assertEqual(config["Input"]["coordinate reference system"], "coordinate_system")
        self.assertEqual(config["Output"]["output filepath"], "output_folder_path")
        self.assertEqual(config["Input"]["output style files"], "output_style_files")

        # Cleanup
        temp_file.close()
        os.remove(temp_file.name)

    # @patch('seat.stressor_receptor_calc.QgsProject')
    # @patch('seat.stressor_receptor_calc.QgsRasterLayer')
    # def test_style_layer(self, mock_QgsRasterLayer, mock_QgsProject):
    #     """
    #     Test the style_layer method.
    #     """
    #     mock_layer = MagicMock()
    #     mock_QgsRasterLayer.return_value = mock_layer
    #     mock_project = MagicMock()
    #     mock_QgsProject.instance.return_value = mock_project

    #     # Mock methods used in the function
    #     mock_layer.loadNamedStyle = MagicMock()
    #     mock_layer.triggerRepaint = MagicMock()
    #     mock_layer.reload = MagicMock()
    #     mock_layer.legendSymbologyItems = MagicMock(return_value=[('range1',), ('range2',)])
    #     self.stressor_receptor_calc.iface.layerTreeView().refreshLayerSymbology = MagicMock()
        
    #     # Call the function with a style path
    #     fpath = "test_fpath.tif"
    #     stylepath = "test_style.qml"
    #     result = self.stressor_receptor_calc.style_layer(fpath, stylepath)

    #     # Assert that the style was applied and layer added to the project
    #     mock_layer.loadNamedStyle.assert_called_once_with(stylepath)
    #     mock_project.addMapLayer.assert_called_once_with(mock_layer)
    #     self.assertEqual(result, ['range1', 'range2'])

    #     # Test the function with an empty style path
    #     mock_layer.reset_mock()
    #     mock_project.reset_mock()
    #     result = self.stressor_receptor_calc.style_layer(fpath, "")

    #     # Assert that the style was not applied
    #     mock_layer.loadNamedStyle.assert_not_called()
    #     self.assertIsNone(result)

def run_all():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestStressorReceptorCalcModule))
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == '__main__':
    run_all()
