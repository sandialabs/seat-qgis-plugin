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
        # stressor_receptor_calc = sr.StressorReceptorCalc(None)  
        # stressor_receptor_calc.dlg = 
        # result_df = stressor_receptor_calc.read_style_files(self.mock_csv_file)

        # Adjusting expected_df to have the same index name as result_df
        # expected_df = pd.DataFrame({'Value': [10, 20, 30]}, index=pd.Index(['Type1', 'Type2', 'Type3'], name='Type'))
        
        # pd.testing.assert_frame_equal(result_df, expected_df)
  

        # mock object created to simulate the behavior of QFileDialog.getOpenFileName.
        mock_filename="C:/Users/sterl/Codes/seat-qgis-plugin/tests/data/structured/probabilities/probabilities.csv"
        mock_getOpenFileName.return_value = (mock_filename, None)

        self.stressor_receptor_calc.select_probabilities_file()
        # checks that the setText method of self.stressor_receptor_calc.dlg.  probabilities_file was called exactly once with mock_filename as its argument. 
        self.stressor_receptor_calc.dlg.probabilities_file.setText.assert_called_once_with(mock_filename)



def run_all():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestStressorReceptorCalcModule))
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == '__main__':
    run_all()
