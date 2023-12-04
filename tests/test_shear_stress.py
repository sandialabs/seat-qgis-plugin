import sys
import os
import unittest
import numpy as np
from qgis.core import QgsApplication
# Import seat
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
from seat import shear_stress_module as ssm


# Mock Interface
class MockIface:
    pass


class TestShearStress(unittest.TestCase):

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
            0.8,  # New Erosion (dev >= 1, nodev < 1)
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
        expected_classification = np.array([ 3.,  2.,  1., 0., -1., -2.,  -3.]) 
        result = ssm.classify_mobility(mobility_parameter_dev, mobility_parameter_nodev)
        np.testing.assert_array_equal(result, expected_classification)

def run_all():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestShearStress))
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == '__main__':
    run_all()
