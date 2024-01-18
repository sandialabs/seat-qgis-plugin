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
from seat import stressor_receptor_calc as sr
# fmt: on


class TestStressorReceptorCalcModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Class method to set up test environment.
        """
        # Other setup code...

        # Create a mock CSV file for testing
        cls.mock_csv_file = 'mock_style_file.csv'
        cls.create_mock_csv(cls.mock_csv_file)

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

    

def run_all():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestStressorReceptorCalcModule))
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == '__main__':
    run_all()
